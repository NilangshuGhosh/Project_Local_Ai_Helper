import os
import time
import warnings
from typing import List, Generator, Dict, Any, Optional

warnings.filterwarnings("ignore")
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from core.config import LLM_MODEL, OLLAMA_BASE_URL, MAX_CONTEXT_CHARS, TOP_K_RETRIEVAL
from core.memory import memory_manager
from core.hybrid_search import hybrid_engine
from core.database import get_session_documents

# Strict factual system prompt with balanced context handling, dialogue memory, and citations
SYSTEM_PROMPT_TEMPLATE = (
    "You are a strict, factual local AI assistant. Answer the user's question BASED on the provided Context below.\n\n"
    "CRITICAL OPERATIONAL RULES:\n"
    "1. Answer BASED on the provided context. Ground every factual claim directly in the context.\n"
    "2. If the context is partially relevant, use it and be explicit about what the context does and does not say.\n"
    "3. Only if there is absolutely nothing related to the question in the context, say so clearly: 'I cannot find that information in the document.'\n"
    "4. Do NOT invent data, formulas, or outside details that are not in the context.\n"
    "5. CITATION REQUIREMENT: After every factual statement or answer, append its source citation tag in brackets, for example: [Source 1] or [Source 2]. Use ONLY the [Source N] tags provided in the context below.\n"
    "6. Keep the answer direct, factually grounded, and concise.\n\n"
    "{history_section}"
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
        raw_source = meta.get("source_file") or meta.get("source", "")
        file_name = meta.get("file_name") or (os.path.basename(raw_source) if raw_source else "Document")
        page = meta.get("page")
        para = meta.get("paragraph_idx")

        # Header tag
        if page is not None and meta.get("file_type") == "pdf":
            header = f"[Source {idx}: {file_name} | Page {page} | Para {para}]"
        elif para is not None:
            header = f"[Source {idx}: {file_name} | Para {para}]"
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
    query: str,
    session_name: Optional[str] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Real-time streaming generator for RAG responses with conversational memory support.
    
    Yields:
        - {"type": "context", "docs": docs, "retrieval_ms": int, "retrieval_query": str}
        - {"type": "token", "token": str}
        - {"type": "done", "total_tokens": int, "elapsed_s": float, "ttft_s": float}
    """
    t_start = time.perf_counter()

    # 1. Formulate retrieval query (lightweight heuristic anaphora resolution if follow-up)
    retrieval_query = query
    if session_name:
        retrieval_query = memory_manager.get_contextual_retrieval_query(session_name, query)

    # 2. Retrieve matching chunks via Hybrid Search (Dense Vector + Lexical BM25 + RRF)
    if session_name:
        docs, ret_metrics = hybrid_engine.search(
            session_name=session_name,
            retriever=retriever,
            query=retrieval_query,
            get_docs_fn=get_session_documents,
            top_k=TOP_K_RETRIEVAL
        )
        retrieval_ms = ret_metrics["total_ms"]
    else:
        t_ret_start = time.perf_counter()
        docs = retriever.invoke(retrieval_query)
        retrieval_ms = round((time.perf_counter() - t_ret_start) * 1000, 1)

    # Yield retrieved context immediately so the caller has document metadata
    yield {
        "type": "context",
        "docs": docs,
        "retrieval_ms": retrieval_ms,
        "retrieval_query": retrieval_query if retrieval_query != query else None
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

    # 3. Build lean, budgeted prompt with optional conversation memory
    formatted_context = format_context_with_budget(docs)
    
    formatted_history = ""
    if session_name:
        formatted_history = memory_manager.get_formatted_history(session_name)

    if formatted_history:
        history_block = f"Recent Conversation (for reference/pronouns):\n{formatted_history}\n\n"
    else:
        history_block = ""

    system_text = SYSTEM_PROMPT_TEMPLATE.format(
        history_section=history_block,
        context=formatted_context
    )

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=query)
    ]

    # 4. Stream tokens from Ollama
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
        ("system", SYSTEM_PROMPT_TEMPLATE.replace("{history_section}", "").replace("{context}", "{context}")),
        ("human", "{input}"),
    ])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)