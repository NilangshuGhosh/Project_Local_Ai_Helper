import os
import sqlite3
from typing import List, Dict, Any, Optional
from core.config import CHAT_HISTORY_DB, DB_DIR


def _get_connection() -> sqlite3.Connection:
    """Returns a connection to the persistent chat history SQLite database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(CHAT_HISTORY_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_chat_history_db() -> None:
    """Initializes the chat history table and indexes if they do not exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                sources_summary TEXT,
                model_used TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_time 
            ON session_messages(session_name, timestamp)
        """)
        conn.commit()


def save_chat_turn(
    session_name: str,
    user_query: str,
    ai_response: str,
    sources_summary: str = "",
    model_used: str = ""
) -> int:
    """
    Persists a single completed Q&A turn to the SQLite database.
    Returns the inserted row ID.
    """
    if not session_name or not user_query:
        return -1

    init_chat_history_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO session_messages (
                session_name, user_query, ai_response, sources_summary, model_used
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            session_name.strip(),
            user_query.strip(),
            ai_response.strip(),
            sources_summary.strip(),
            model_used.strip()
        ))
        conn.commit()
        return cursor.lastrowid


def get_recent_session_turns(session_name: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent N turns for a session (used for session resumption on connect).
    Returned in chronological order (oldest to newest).
    """
    if not session_name or not os.path.exists(CHAT_HISTORY_DB):
        return []

    init_chat_history_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, session_name, timestamp, user_query, ai_response, sources_summary, model_used
            FROM session_messages
            WHERE session_name = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        """, (session_name.strip(), limit))
        rows = cursor.fetchall()
        
        # Reverse to return in chronological order
        return [dict(r) for r in reversed(rows)]


def get_all_session_turns(session_name: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieves paginated conversation history for display or export.
    Returned in chronological order.
    """
    if not session_name or not os.path.exists(CHAT_HISTORY_DB):
        return []

    init_chat_history_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, session_name, timestamp, user_query, ai_response, sources_summary, model_used
            FROM session_messages
            WHERE session_name = ?
            ORDER BY timestamp ASC, id ASC
            LIMIT ? OFFSET ?
        """, (session_name.strip(), limit, offset))
        return [dict(r) for r in cursor.fetchall()]


def get_session_message_count(session_name: str) -> int:
    """Returns the total number of saved messages for a given session."""
    if not session_name or not os.path.exists(CHAT_HISTORY_DB):
        return 0

    init_chat_history_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM session_messages WHERE session_name = ?", (session_name.strip(),))
        row = cursor.fetchone()
        return row[0] if row else 0


def delete_session_chat_history(session_name: str) -> int:
    """
    Deletes only the chat messages for a specific session without affecting document vectors.
    Returns the number of deleted rows.
    """
    if not session_name or not os.path.exists(CHAT_HISTORY_DB):
        return 0

    init_chat_history_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM session_messages WHERE session_name = ?", (session_name.strip(),))
        conn.commit()
        return cursor.rowcount


def delete_all_chat_history() -> int:
    """Deletes all chat history rows across all sessions."""
    if not os.path.exists(CHAT_HISTORY_DB):
        return 0

    init_chat_history_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM session_messages")
        conn.commit()
        return cursor.rowcount

