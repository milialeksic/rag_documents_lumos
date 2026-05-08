import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
load_dotenv()
vectorstore_path = "vectorstore/"

def load_vectorstore():
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vectorstore not found at {vectorstore_path}.")
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=vectorstore_path, embedding_function=embeddings, collection_name="knowledge_base")

# def build_retriever(vectorstore):
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#     base_retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 4}
#     )
#     return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
from langchain_community.retrievers import BM25Retriever


def build_retriever(vectorstore, documents):
    # Vector retriever
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4
    
    # Combine both
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever
from ingest import load_documents, save_documents, load_saved_documents

def ask(question):
    vectorstore = load_vectorstore()
    
    documents = load_saved_documents()
    retriever = build_retriever(vectorstore,documents)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant for Lumos student organization.\n"
        "You will be given several chunks from different pages of documents.\n"
        "Read ALL chunks carefully and connect information across them to answer the question.\n"
        "For example, if one chunk says 'Cooking with Claude' and another says 'Presented by Michael Bösch',\n"
        "you should connect these and answer that Michael Bösch presented Cooking with Claude.\n"
        "Always cite which document and page your answer comes from.\n"
        "If you cannot find the answer in the chunks, say so clearly.\n\n"
        "Context chunks:\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- ANSWER ---")
    answer = chain.invoke(question)
    print(answer)

    print("\n--- SOURCES ---")
    docs = retriever.invoke(question)
    seen = set()
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', '')
        ref = f"{source} (page {page})" if page != '' else source
        if ref not in seen:
            seen.add(ref)
            print(f"- {ref}")

def ask_with_sources(question):
    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant for Lumos student organization.\n"
        "Read ALL chunks carefully and connect information across them to answer the question.\n"
        "Read the context carefully. Do not mix up different people's names and roles.\n"
        "Each person has their own name and description — keep them separate.\n"
        "If multiple people are mentioned together, list all of them in your answer.\n"
        "Always cite which document and page your answer comes from.\n"
        "If you cannot find the answer, say so clearly.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', '')
        ref = f"{source} (page {page})" if page != '' else source
        if ref not in seen:
            seen.add(ref)
            sources.append(ref)

    return answer, sources

if __name__ == "__main__":
    question = input("Ask a question about Lumos: ")
    ask(question)