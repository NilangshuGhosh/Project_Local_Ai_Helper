import os
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from core.config import DB_DIR, EMBEDDING_MODEL

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

def get_chroma_client():
    os.makedirs(DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=DB_DIR)

def list_sessions():
    """Returns a list of dicts: [{'name': col.name, 'count': col.count()}]"""
    client = get_chroma_client()
    collections = client.list_collections()
    return [{"name": col.name, "count": col.count()} for col in collections]

def delete_session(session_name: str):
    client = get_chroma_client()
    try:
        client.delete_collection(session_name)
        return True
    except ValueError:
        return False

def ingest_document(file_path: str, session_name: str, splits):
    """Stores document splits under a named collection."""
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=session_name,
    )

def get_retriever(session_name: str, k: int = 5):
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=session_name,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})