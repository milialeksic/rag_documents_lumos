import os
import pickle
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from functools import lru_cache



load_dotenv()

VECTORSTORE_PATH = "vectorstore/"
DOCUMENTS_PATH = "vectorstore/documents.pkl"

# ── Load vectorstore ──────────────────────────────────────────────────────────

def load_vectorstore():
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(f"Vectorstore not found. Run ingest.py first.")
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
        collection_name="knowledge_base"
    )

# ── Load saved documents for BM25 ────────────────────────────────────────────

def load_saved_documents(path=DOCUMENTS_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Build hybrid retriever ────────────────────────────────────────────────────

def build_retriever(vectorstore, documents):
    # Vector retriever — finds semantically similar chunks
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # BM25 retriever — finds exact keyword matches
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 6

    # Combine both with equal weight
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

# ── Expand retrieved chunks with neighbors (parent context simulation) ────────

def expand_with_neighbors(docs, all_documents, window=2):
    """
    For each retrieved chunk, fetch surrounding pages within a window.
    window=2 means fetch 2 pages before and 2 pages after each retrieved chunk.
    """
    expanded = []
    seen_ids = set()

    # Build lookup: (source, page) -> document
    doc_map = {}
    for doc in all_documents:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        if source and page is not None:
            doc_map[(source, page)] = doc

    for doc in docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        if page is None:
            # No page metadata — just add as-is
            key = (source, None)
            if key not in seen_ids:
                seen_ids.add(key)
                expanded.append(doc)
            continue

        # Add window of pages around current chunk
        for offset in range(-window, window + 1):
            neighbor_key = (source, page + offset)
            if neighbor_key in doc_map and neighbor_key not in seen_ids:
                seen_ids.add(neighbor_key)
                expanded.append(doc_map[neighbor_key])

    return expanded

# ── Prompt ────────────────────────────────────────────────────────────────────

def build_prompt():
    return PromptTemplate.from_template(
        "You are a helpful assistant for Lumos student organization.\n"
        "You will be given several chunks from different pages of documents.\n\n"
        "Rules:\n"
        "1. Read ALL chunks carefully before answering.\n"
        "2. Use every piece of information available, even short descriptions or labels.\n"
        "3. Connect information across different chunks and pages to form a complete answer.\n"
        "4. If multiple people are mentioned, list ALL of them with their descriptions.\n"
        "5. Never say there is 'no further information' if there is any description in the chunks.\n"
        "6. Do NOT include source references inside your answer text.\n"
        "7. If the answer truly cannot be found in any chunk, say so clearly.\n\n"
        "At the end of your answer, add a line starting with 'SOURCES_USED:' followed by "
        "a comma separated list of ONLY the source filenames you actually used.\n"
        "Example: SOURCES_USED: filename1.pdf, filename2.md\n\n"
        "Context chunks:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

# ── Ask ───────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_cached_documents():
    return load_saved_documents()

@lru_cache(maxsize=1)
def get_cached_retriever():
    vectorstore = load_vectorstore()
    documents = get_cached_documents()
    return build_retriever(vectorstore, documents)


def ask(question):
    # vectorstore = load_vectorstore()
    all_documents = get_cached_documents()
    retriever = get_cached_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Retrieve and expand with neighboring pages
    retrieved_docs = retriever.invoke(question)
    expanded_docs = expand_with_neighbors(retrieved_docs, all_documents)

    # Format context
    context = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in expanded_docs
    ])

    # Build and run chain
    prompt = build_prompt()
    llm_chain = prompt | llm | StrOutputParser()

    print("\n--- ANSWER ---")
    answer = llm_chain.invoke({"context": context, "question": question})
    print(answer)

    print("\n--- SOURCES ---")
    seen = set()
    for doc in expanded_docs:
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', '')
        ref = f"{source} (page {page})" if page != '' else source
        if ref not in seen:
            seen.add(ref)
            print(f"- {ref}")

def ask_with_sources(question):
    all_documents = get_cached_documents()
    retriever = get_cached_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retrieved_docs = retriever.invoke(question)
    expanded_docs = expand_with_neighbors(retrieved_docs, all_documents, window=1)

    context = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in expanded_docs
    ])

    prompt = build_prompt()
    llm_chain = prompt | llm | StrOutputParser()
    full_response = llm_chain.invoke({"context": context, "question": question})

    # Split answer and sources
    if "SOURCES_USED:" in full_response:
        parts = full_response.split("SOURCES_USED:")
        answer = parts[0].strip()
        sources_line = parts[1].strip()
        # Parse source filenames
        raw_sources = [s.strip() for s in sources_line.split(",") if s.strip()]
        # Match to actual file paths
        sources = []
        for raw in raw_sources:
            for doc in expanded_docs:
                doc_source = doc.metadata.get("source", "")
                if raw.lower() in doc_source.lower() or doc_source.lower().endswith(raw.lower()):
                    ref = doc_source
                    if ref not in sources:
                        sources.append(ref)
                    break
    else:
        # Fallback to retrieved docs if LLM didn't follow format
        answer = full_response
        seen = set()
        sources = []
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in seen:
                seen.add(source)
                sources.append(source)
    # At the end of ask_with_sources, before return:
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.replace("[]", "").replace("[ ]", "").strip()

    return answer, sources


if __name__ == "__main__":
    question = input("Ask a question about Lumos: ")
    ask(question)