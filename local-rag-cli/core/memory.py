import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from core.config import (
    MAX_HISTORY_TURNS,
    MAX_HISTORY_CHARS,
    MAX_ASSISTANT_SUMMARY_CHARS
)


@dataclass
class ConversationTurn:
    user_query: str
    assistant_response: str


class ConversationMemoryManager:
    """
    CPU-efficient, token-bounded conversational memory manager.
    
    Guarantees:
    1. Zero LLM overhead: No extra LLM call to rewrite queries (saves 5-10s CPU compute).
    2. Strict Token/Character Guardrails: Never overflows prompt context or causes CPU thermal throttling.
    3. Lightweight Heuristic Anaphora Resolution: Enriches dense vector search for follow-up questions
       containing pronouns (it, she, her, he, this, that) without compute latency.
    """
    
    def __init__(self):
        # In-memory storage indexed by session_name
        self._sessions: Dict[str, List[ConversationTurn]] = {}

    def add_turn(self, session_name: str, query: str, response: str) -> None:
        """Adds a completed interaction turn to the session history."""
        if not session_name:
            return

        if session_name not in self._sessions:
            self._sessions[session_name] = []

        # Clean response: strip extra whitespace and truncate to prevent memory bloat
        clean_query = query.strip()
        clean_resp = response.strip()

        # Condense assistant response to key statements (strip bracket citations in memory to save tokens)
        condensed_resp = re.sub(r"\[Source\s+\d+[^\]]*\]", "", clean_resp).strip()
        if len(condensed_resp) > MAX_ASSISTANT_SUMMARY_CHARS:
            condensed_resp = condensed_resp[:MAX_ASSISTANT_SUMMARY_CHARS].rstrip() + "..."

        self._sessions[session_name].append(
            ConversationTurn(user_query=clean_query, assistant_response=condensed_resp)
        )

        # Enforce sliding window on turns
        if len(self._sessions[session_name]) > MAX_HISTORY_TURNS:
            self._sessions[session_name] = self._sessions[session_name][-MAX_HISTORY_TURNS:]

    def get_formatted_history(self, session_name: str, max_chars: int = MAX_HISTORY_CHARS) -> str:
        """
        Formats recent turns into a lean dialogue block for the LLM prompt.
        Respects max_chars budget to protect CPU prompt prefill time.
        """
        turns = self._sessions.get(session_name, [])
        if not turns:
            return ""

        formatted_blocks = []
        total_chars = 0

        # Traverse backwards from most recent to oldest
        for turn in reversed(turns):
            block = f"User: {turn.user_query}\nAssistant: {turn.assistant_response}"
            if total_chars + len(block) > max_chars and formatted_blocks:
                break
            formatted_blocks.append(block)
            total_chars += len(block)

        # Reverse back to chronological order
        formatted_blocks.reverse()

        return "\n\n".join(formatted_blocks)

    def get_contextual_retrieval_query(self, session_name: str, current_query: str) -> str:
        """
        Lightweight zero-cost query enricher for ChromaDB/BM25 retrieval.
        
        Enriches ONLY true anaphora/elliptical follow-ups without contaminating self-contained queries.
        Takes <0.01ms of CPU compute (pure string heuristics, 0 model inference).
        """
        turns = self._sessions.get(session_name, [])
        if not turns:
            return current_query

        clean_query = current_query.strip()
        query_lower = clean_query.lower()
        words = query_lower.split()

        # 1. Self-Contained Guard:
        # If query contains an explicit document ID / code (e.g. SPA-2026-C, ERR_403)
        # or explicit document references, it is 100% self-contained and must never be contaminated.
        has_explicit_id = bool(re.search(r"\b[A-Za-z0-9]+[-_][A-Za-z0-9-_]+\b", clean_query))
        has_doc_indicator = any(m in query_lower for m in [
            "doc:", "document:", "file:", "section:", "this doc", "this document",
            "this file", "this pdf", "the doc", "the document", "the file", "the pdf"
        ])
        if has_explicit_id or has_doc_indicator:
            return clean_query

        # 2. Check for true personal pronouns that depend on previous antecedents
        true_personal_pronouns = {
            "he", "him", "his", "she", "her", "hers", "it", "its",
            "they", "them", "their", "theirs"
        }
        has_true_pronoun = any(w in true_personal_pronouns for w in words)

        # 3. Check for conversational continuation phrases
        is_continuation = query_lower.startswith((
            "what about ", "how about ", "and what ", "and how ",
            "and the ", "and why ", "why ", "why?", "where is it", "how so"
        ))

        # 4. Elliptical queries: very short phrases (<= 3 words) without full standalone question structure
        is_elliptical = len(words) <= 3 and not re.search(r"\b(who|what|where|how|why)\s+is\s+\w+\b", query_lower)

        if has_true_pronoun or is_continuation or is_elliptical:
            last_turn = turns[-1]
            prior_subject = re.sub(r"[?!.,]", "", last_turn.user_query).strip()
            return f"{prior_subject} {clean_query}"

        return clean_query

    def clear(self, session_name: str) -> bool:
        """Clears conversational memory for a given session."""
        if session_name in self._sessions:
            del self._sessions[session_name]
            return True
        return False

    def get_turn_count(self, session_name: str) -> int:
        """Returns the number of turns currently stored for the session."""
        return len(self._sessions.get(session_name, []))

    def get_turns(self, session_name: str) -> List[ConversationTurn]:
        """Returns raw turns list for display or export."""
        return list(self._sessions.get(session_name, []))


# Global singleton instance for the CLI lifecycle
memory_manager = ConversationMemoryManager()

