import os
import gc
import stat
import time
import shutil
import sqlite3
import re
import warnings
from typing import List, Dict, Any, Optional

# Suppress minor deprecation warning from langchain_community
warnings.filterwarnings("ignore", category=DeprecationWarning)

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from core.config import DB_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL

# Reusable embedding instance with configurable base_url
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)


def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a PersistentClient for ChromaDB, ensuring DB_DIR exists."""
    os.makedirs(DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=DB_DIR)


def _remove_readonly(func, path, excinfo):
    """Windows error handler for shutil.rmtree to remove read-only flags and retry."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def cleanup_orphaned_segments() -> int:
    """
    Physically removes orphaned segment directories from chroma_db on disk
    and reclaims SQLite disk space via VACUUM.
    
    Returns the number of deleted orphan directories.
    """
    db_file = os.path.join(DB_DIR, "chroma.sqlite3")
    if not os.path.exists(db_file):
        return 0

    try:
        chromadb.PersistentClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()

    active_segments = set()
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("SELECT id FROM segments")
        rows = cur.fetchall()
        active_segments = {str(row[0]) for row in rows}
        conn.close()
    except Exception:
        return 0

    deleted_count = 0
    if os.path.exists(DB_DIR):
        for item in os.listdir(DB_DIR):
            item_path = os.path.join(DB_DIR, item)
            if os.path.isdir(item_path):
                if item not in active_segments:
                    for attempt in range(4):
                        try:
                            gc.collect()
                            shutil.rmtree(item_path, onerror=_remove_readonly)
                            deleted_count += 1
                            break
                        except Exception:
                            time.sleep(0.15)

    try:
        conn = sqlite3.connect(db_file)
        conn.execute("VACUUM")
        conn.close()
    except Exception:
        pass

    return deleted_count


def get_session_file_summary(session_name: str) -> List[Dict[str, Any]]:
    """
    Returns a breakdown of files contained within a session collection:
    [{'file_name': 'manual.pdf', 'chunk_count': 12, 'file_type': 'pdf'}]
    """
    client = get_chroma_client()
    try:
        col = client.get_collection(session_name)
        data = col.get(include=["metadatas"])
        summary = {}
        for meta in data.get("metadatas", []):
            if not meta:
                continue
            fname = meta.get("file_name", "unknown")
            ftype = meta.get("file_type", "unknown")
            if fname not in summary:
                summary[fname] = {"file_name": fname, "chunk_count": 0, "file_type": ftype}
            summary[fname]["chunk_count"] += 1
        return list(summary.values())
    except Exception:
        return []
    finally:
        try:
            client.close()
        except Exception:
            pass


def list_sessions() -> List[Dict[str, Any]]:
    """
    Returns a list of dicts:
    [{'name': col.name, 'count': col.count(), 'files': file_count}]
    """
    client = get_chroma_client()
    try:
        collections = client.list_collections()
        sessions = []
        for col in collections:
            count = col.count()
            # Inspect metadata to get distinct file count
            files_count = 0
            try:
                data = col.get(include=["metadatas"])
                distinct_files = {m.get("file_name") for m in data.get("metadatas", []) if m and "file_name" in m}
                files_count = len(distinct_files)
            except Exception:
                pass

            sessions.append({
                "name": col.name,
                "count": count,
                "file_count": files_count
            })
        return sessions
    finally:
        try:
            client.close()
        except Exception:
            pass


def delete_session(session_name: str) -> bool:
    """
    Deletes a session from ChromaDB logically AND physically purges its segment folders
    from the filesystem, preventing ghost data accumulation.
    """
    client = get_chroma_client()
    try:
        existing = [c.name for c in client.list_collections()]
        if session_name not in existing:
            return False

        client.delete_collection(session_name)
        
        try:
            client.clear_system_cache()
            client.close()
        except Exception:
            pass
        del client
        gc.collect()

        cleanup_orphaned_segments()
        return True
    except Exception:
        cleanup_orphaned_segments()
        return False


def _sanitize_id(text: str) -> str:
    """Sanitizes strings to be safe for vector ID usage."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', text)


def ingest_document(file_path: str, session_name: str, splits) -> int:
    """
    Stores document splits under a named collection atomically with deterministic IDs.
    If an error occurs mid-ingestion, rolls back partially created data
    so the vector database does not get corrupted.
    
    Returns the number of chunks successfully indexed.
    """
    client = get_chroma_client()
    existing_collections = [c.name for c in client.list_collections()]
    is_new_collection = session_name not in existing_collections

    # Generate traceable, deterministic chunk IDs
    file_base = os.path.basename(file_path)
    chunk_ids = [
        f"{_sanitize_id(session_name)}_{_sanitize_id(doc.metadata.get('file_name', file_base))}_chunk_{doc.metadata.get('chunk_index', i+1)}"
        for i, doc in enumerate(splits)
    ]

    try:
        Chroma.from_documents(
            documents=splits,
            ids=chunk_ids,
            embedding=embeddings,
            persist_directory=DB_DIR,
            collection_name=session_name,
        )
        return len(splits)
    except Exception as e:
        if is_new_collection:
            try:
                client.delete_collection(session_name)
                client.clear_system_cache()
                client.close()
            except Exception:
                pass
            cleanup_orphaned_segments()
        raise RuntimeError(f"Ingestion failed: {e}") from e


def get_retriever(session_name: str, k: int = 5):
    """Returns a LangChain retriever for the specified session collection."""
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=session_name,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_database_stats() -> Dict[str, Any]:
    """
    Calculates physical and logical storage statistics for user visibility.
    """
    sessions = list_sessions()
    total_chunks = sum(s["count"] for s in sessions)
    
    total_size_bytes = 0
    file_count = 0
    if os.path.exists(DB_DIR):
        for root, _, files in os.walk(DB_DIR):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total_size_bytes += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    pass

    return {
        "session_count": len(sessions),
        "total_chunks": total_chunks,
        "size_mb": round(total_size_bytes / (1024 * 1024), 2),
        "file_count": file_count,
        "db_dir": DB_DIR
    }