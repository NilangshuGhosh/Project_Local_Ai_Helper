import time
import warnings
from typing import List, Generator, Dict, Any

warnings.filterwarnings("ignore")
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from core.config import LLM_MODEL, OLLAMA_BASE_URL, MAX_CONTEXT_CHARS

# Strict system prompt ensuring zero hallucination / zero extrapolation
SYSTEM_PROMPT_TEMPLATE = (
    "You are a strict, factual local AI assistant. Answer the user's question using ONLY the provided Context below.\n\n"
    "CRITICAL OPERATIONAL RULES:\n"
    "1. Rely strictly on the explicit facts directly mentioned in the Context.\n"
    "2. If the answer cannot be found in the context, or is only partially mentioned, answer ONLY what is directly stated and explicitly state: 'I cannot find further information about that in the document.'\n"
    "3. NEVER use outside or prior training knowledge. Do NOT extrapolate, speculate, or invent formulas, facts, or details not directly present in the context.\n"
    "4. Keep the answer direct, factually grounded, and concise.\n\n"
    "Context:\n{context}"
)


def get_chat_llm() -> ChatOllama:
    """Returns a ChatOllama instance configured for low-temperature factual generation."""
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.0,
        base_url=OLLAMA_BASE_URL,
        keep_alive=3600
    )


def format_context_with_budget(docs: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Formats retrieved document chunks with source metadata headers while enforcing
    a strict context character budget to protect laptop RAM/VRAM and ensure fast inference.
    """
    if not docs:
        return "No relevant context found."

    context_blocks = []
    total_chars = 0

    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        file_name = meta.get("file_name", "Unknown File")
        page = meta.get("page")
        para = meta.get("paragraph_idx")

        # Header tag
        if page is not None and meta.get("file_type") == "pdf":
            header = f"[Source {idx}: {file_name} | Page {page} | ¶{para}]"
        elif para is not None:
            header = f"[Source {idx}: {file_name} | ¶{para}]"
        else:
            header = f"[Source {idx}: {file_name}]"

        content = doc.page_content.strip()
        block = f"{header}\n{content}"

        if total_chars + len(block) > max_chars and context_blocks:
            # Stop appending if context budget is exceeded
            break

        context_blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(context_blocks)


def stream_rag_response(
    retriever,
    query: str
) -> Generator[Dict[str, Any], None, None]:
    """
    Real-time streaming generator for RAG responses.
    
    Yields:
        - {"type": "context", "docs": docs, "retrieval_ms": int}
        - {"type": "token", "token": str}
        - {"type": "done", "total_tokens": int, "elapsed_s": float, "ttft_s": float}
    """
    t_start = time.perf_counter()

    # 1. Retrieve matching chunks from vector database
    t_ret_start = time.perf_counter()
    docs = retriever.invoke(query)
    retrieval_ms = round((time.perf_counter() - t_ret_start) * 1000, 1)

    # Yield retrieved context immediately so the caller has document metadata
    yield {
        "type": "context",
        "docs": docs,
        "retrieval_ms": retrieval_ms
    }

    if not docs:
        fallback_msg = "I cannot find that information in the document."
        yield {"type": "token", "token": fallback_msg}
        yield {
            "type": "done",
            "total_tokens": len(fallback_msg.split()),
            "elapsed_s": round(time.perf_counter() - t_start, 2),
            "ttft_s": 0.0
        }
        return

    # 2. Build lean, budgeted prompt
    formatted_context = format_context_with_budget(docs)
    system_text = SYSTEM_PROMPT_TEMPLATE.format(context=formatted_context)

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=query)
    ]

    # 3. Stream tokens from Ollama
    llm = get_chat_llm()
    token_count = 0
    t_first_token = None

    try:
        for chunk in llm.stream(messages):
            if t_first_token is None:
                t_first_token = time.perf_counter()

            token_text = chunk.content
            if token_text:
                token_count += 1
                yield {"type": "token", "token": token_text}
    except Exception as e:
        yield {"type": "error", "error": str(e)}
        return

    total_elapsed = round(time.perf_counter() - t_start, 2)
    ttft_s = round((t_first_token - t_start) if t_first_token else total_elapsed, 2)

    yield {
        "type": "done",
        "total_tokens": token_count,
        "elapsed_s": total_elapsed,
        "ttft_s": ttft_s
    }


def build_chat_chain(retriever):
    """
    Legacy chain builder maintained for backwards compatibility.
    Prefer stream_rag_response for real-time interactive chat.
    """
    llm = get_chat_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE.replace("{context}", "{context}")),
        ("human", "{input}"),
    ])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)