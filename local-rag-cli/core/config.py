import os

DB_DIR = os.path.abspath("./chroma_db")
CHAT_HISTORY_DB = os.path.join(DB_DIR, "chat_history.sqlite3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"

# Dynamic Model Registry
SUPPORTED_MODELS = {
    "qwen2.5:1.5b": {
        "name": "Qwen 2.5 (1.5B)",
        "tag": "qwen2.5:1.5b",
        "description": "Ultra Fast / Low TTFT (Recommended for 8GB RAM laptops)"
    },
    "phi3": {
        "name": "Phi-3 (3.8B)",
        "tag": "phi3",
        "description": "High Quality / Standard TTFT"
    }
}

# Default active model (persists during process lifetime)
_ACTIVE_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "qwen2.5:1.5b")


def get_active_model() -> str:
    """Returns the currently active LLM model tag."""
    global _ACTIVE_LLM_MODEL
    return _ACTIVE_LLM_MODEL


def set_active_model(model_tag: str) -> bool:
    """Switches the active LLM model dynamically at runtime."""
    global _ACTIVE_LLM_MODEL
    if model_tag in SUPPORTED_MODELS or model_tag:
        _ACTIVE_LLM_MODEL = model_tag
        return True
    return False


# For backwards compatibility with static references
LLM_MODEL = _ACTIVE_LLM_MODEL

# Standard text & PDF chunking parameters (tuned for consumer laptops)
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Code chunking parameters (slightly larger to preserve full functions)
CODE_CHUNK_SIZE = 800
CODE_CHUNK_OVERLAP = 120

# Performance & Context Guardrails for Consumer Laptops
TOP_K_RETRIEVAL = 4
MAX_CONTEXT_CHARS = 3000

# Conversational Memory Guardrails (Protects CPU from thermal throttling & prefill latency)
MAX_HISTORY_TURNS = 3             # Keep at most last 3 interaction turns (user-assistant pairs)
MAX_HISTORY_CHARS = 750           # Hard ceiling for formatted dialogue history in prompt
MAX_ASSISTANT_SUMMARY_CHARS = 200 # Compact representation for previous AI replies in memory