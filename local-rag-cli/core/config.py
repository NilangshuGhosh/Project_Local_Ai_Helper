import os

DB_DIR = os.path.abspath("./chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 125