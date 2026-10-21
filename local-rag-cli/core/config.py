import os

DB_DIR = os.path.abspath("./chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3"

# Standard text & PDF chunking parameters (tuned for consumer laptops)
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Code chunking parameters (slightly larger to preserve full functions)
CODE_CHUNK_SIZE = 800
CODE_CHUNK_OVERLAP = 120