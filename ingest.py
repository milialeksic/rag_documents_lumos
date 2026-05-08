import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader,PyMuPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, CSVLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from collections import Counter
import pdfplumber
from langchain_core.documents import Document
load_dotenv()

data_path = "data/"
vectorstore_path = "vectorstore/"


from pptx import Presentation


import pickle

def save_documents(documents, path="vectorstore/documents.pkl"):
    with open(path, "wb") as f:
        pickle.dump(documents, f)

def load_saved_documents(path="vectorstore/documents.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def extract_pptx_text(filepath):
    prs = Presentation(filepath)
    documents = []
    for i, slide in enumerate(prs.slides):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text.strip())
        if slide_text:
            documents.append(Document(
                page_content="\n".join(slide_text),
                metadata={"source": filepath, "page": i}
            ))
    return documents

def load_pdf_with_pdfplumber(filepath):
    documents = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={"source": filepath, "page": i}
                ))
    return documents


def load_documents():
    all_documents = []
    file_count = 0
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".txt"):
                    loader = TextLoader(file_path)
                    docs = loader.load()
                elif file.endswith(".pdf"):
                    docs = load_pdf_with_pdfplumber(file_path)
                elif file.endswith(".pptx"):
                    docs = extract_pptx_text(file_path)
                elif file.endswith(".docx"):
                    loader = UnstructuredWordDocumentLoader(file_path)
                    docs = loader.load()
                elif file.endswith(".csv"):
                    loader = CSVLoader(file_path)
                    docs = loader.load()
                elif file.endswith(".xlsx"):
                    loader = UnstructuredExcelLoader(file_path)
                    docs = loader.load()
                else:
                    print(f"Skipping unsupported file: {file}")
                    continue

                file_count += 1
                print(f"  {len(docs)} pages — {file}")
                all_documents.extend(docs)

            except Exception as e:
                print(f"Could not load {file}: {e}")

    print(f"\nLoaded {file_count} files ({len(all_documents)} pages total)")
    return all_documents

def ingest_documents():
    
    documents = load_documents()
    
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks")

    print("Embedding and storing chunks in vector store")

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="knowledge_base", persist_directory=vectorstore_path)
    print("Vector store saved")
    save_documents(chunks)

if __name__ == "__main__":
    ingest_documents()