import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split(file_path: str):
    """Loads a PDF, code, or text file and splits it into chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        # Handles .txt, .cpp, .py, .json, etc.
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)