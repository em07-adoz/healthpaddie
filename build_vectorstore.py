# build_vectorstore.py
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS


DATA_DIR = "C:\\Users\\OLAMIDE\\Downloads\\DHA - Copy-20251122T192346Z-1-001\\DHA - Copy\\health_data"
VECTORSTORE_DIR = "./vectorstore"

def load_documents():
    docs = []

    # PDF files
    pdf_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs.extend(pdf_loader.load())

    # TXT files
    txt_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    docs.extend(txt_loader.load())

    if not docs:
        print("⚠ No documents found in 'health_data/'. Add PDFs or TXT files first.")
    else:
        print(f"✅ Loaded {len(docs)} documents.")
    return docs

def create_vectorstore():
    docs = load_documents()
    if not docs:
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print("✅ Vectorstore saved to './vectorstore'")

if __name__ == "__main__":
    create_vectorstore()
