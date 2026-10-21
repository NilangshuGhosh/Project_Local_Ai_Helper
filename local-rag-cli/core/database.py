import os
import gc
import stat
import time
import shutil
import sqlite3
import re
import hashlib
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


def _vacuum_and_checkpoint():
    """Flushes SQLite WAL logs and reclaims freed disk space via VACUUM."""
    db_file = os.path.join(DB_DIR, "chroma.sqlite3")
    if not os.path.exists(db_file):
        return
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        conn.close()
    except Exception:
        pass


def cleanup_orphaned_segments() -> int:
    """
    Physically removes orphaned segment directories from chroma_db on disk
    and flushes/vacuums SQLite storage.
    
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

    _vacuum_and_checkpoint()
    return deleted_count


def _sanitize_id(text: str) -> str:
    """Sanitizes strings to be safe for vector ID usage."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', text)


def _path_hash(path: str) -> str:
    """Creates a short 8-char hash of full path to uniquely identify identical filenames across directories."""
    norm = os.path.abspath(path).replace("\\", "/").lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:8]


def get_session_file_summary(session_name: str) -> List[Dict[str, Any]]:
    """
    Returns a breakdown of distinct files contained within a session collection.
    Differentiates files with the same basename from different folder locations.
    """
    client = get_chroma_client()
    try:
        col = client.get_collection(session_name)
        data = col.get(include=["metadatas"])
        
        # Group by source_file (unique absolute path)
        files_by_path = {}
        for meta in data.get("metadatas", []):
            if not meta:
                continue
            src_path = meta.get("source_file", meta.get("file_name", "unknown"))
            fname = meta.get("file_name", os.path.basename(src_path))
            ftype = meta.get("file_type", "unknown")
            
            if src_path not in files_by_path:
                files_by_path[src_path] = {
                    "file_name": fname,
                    "source_file": src_path,
                    "file_type": ftype,
                    "chunk_count": 0,
                    "pages": set(),
                }
            files_by_path[src_path]["chunk_count"] += 1
            if "page" in meta:
                files_by_path[src_path]["pages"].add(meta["page"])

        # Format display names (disambiguate if duplicate basenames exist)
        basename_counts = {}
        for item in files_by_path.values():
            bn = item["file_name"]
            basename_counts[bn] = basename_counts.get(bn, 0) + 1

        results = []
        for src_path, item in files_by_path.items():
            if basename_counts[item["file_name"]] > 1:
                # Add parent folder name for disambiguation
                parent_dir = os.path.basename(os.path.dirname(src_path))
                display = f"{item['file_name']} [{parent_dir}]"
            else:
                display = item["file_name"]

            results.append({
                "file_name": item["file_name"],
                "source_file": src_path,
                "display_name": display,
                "file_type": item["file_type"],
                "chunk_count": item["chunk_count"],
                "total_pages": len(item["pages"]) if item["pages"] else 1
            })

        return sorted(results, key=lambda x: x["display_name"].lower())
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
    [{'name': col.name, 'count': col.count(), 'file_count': int}]
    """
    client = get_chroma_client()
    try:
        collections = client.list_collections()
        sessions = []
        for col in collections:
            count = col.count()
            files_count = 0
            try:
                data = col.get(include=["metadatas"])
                distinct_files = {m.get("source_file", m.get("file_name")) for m in data.get("metadatas", []) if m}
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
    Deletes an entire session from ChromaDB logically AND physically purges its segment folders
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


def delete_file_from_session(session_name: str, file_identifier: str, by_path: bool = True) -> Dict[str, Any]:
    """
    Selectively removes a specific document from a multi-file session without deleting the session.
    
    Args:
        session_name: The session collection name.
        file_identifier: The absolute source_file path or file_name to remove.
        by_path: If True, matches against metadata 'source_file' (avoids collisions).
                 If False, matches against metadata 'file_name'.
    
    Returns:
        Dict with deleted_chunks, remaining_chunks, session_deleted (if session became empty).
    """
    client = get_chroma_client()
    try:
        col = client.get_collection(session_name)
        filter_key = "source_file" if by_path else "file_name"
        
        # Check matching chunks count
        existing = col.get(where={filter_key: file_identifier})
        matching_ids = existing.get("ids", [])
        
        if not matching_ids:
            # Fallback check against the other key
            alt_key = "file_name" if by_path else "source_file"
            existing = col.get(where={alt_key: file_identifier})
            matching_ids = existing.get("ids", [])
            filter_key = alt_key

        if not matching_ids:
            return {
                "success": False,
                "deleted_chunks": 0,
                "remaining_chunks": col.count(),
                "session_deleted": False,
                "error": f"No chunks found matching {file_identifier}"
            }

        col.delete(ids=matching_ids)
        remaining = col.count()
        session_deleted = False

        # If no chunks remain in the entire collection, purge the session collection
        if remaining == 0:
            client.delete_collection(session_name)
            session_deleted = True

        try:
            client.clear_system_cache()
            client.close()
        except Exception:
            pass
        del client
        gc.collect()

        if session_deleted:
            cleanup_orphaned_segments()
        else:
            _vacuum_and_checkpoint()

        return {
            "success": True,
            "deleted_chunks": len(matching_ids),
            "remaining_chunks": remaining,
            "session_deleted": session_deleted,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "deleted_chunks": 0,
            "remaining_chunks": -1,
            "session_deleted": False,
            "error": str(e)
        }


def ingest_document(file_path: str, session_name: str, splits) -> Dict[str, Any]:
    """
    Stores document splits under a named collection atomically with deterministic IDs.
    
    Smart Upsert Behavior:
    If this exact file (by source_file) was already ingested in the session, its old chunks
    are purged before inserting the new ones, preventing duplicate or conflicting chunks.
    
    Handles identical filenames across different directories using path hashing.
    
    Returns:
        Dict with status ('created' or 'updated'), chunk_count, replaced_chunks.
    """
    client = get_chroma_client()
    existing_collections = [c.name for c in client.list_collections()]
    is_new_collection = session_name not in existing_collections

    norm_path = os.path.abspath(file_path).replace("\\", "/")
    path_sig = _path_hash(norm_path)
    file_base = os.path.basename(file_path)

    # Deterministic, collision-free chunk IDs
    chunk_ids = [
        f"{_sanitize_id(session_name)}_{_sanitize_id(file_base)}_{path_sig}_c{doc.metadata.get('chunk_index', i+1)}"
        for i, doc in enumerate(splits)
    ]

    replaced_chunks = 0
    status = "created"

    try:
        # If collection exists, check if this file was already ingested and purge stale chunks
        if not is_new_collection:
            col = client.get_collection(session_name)
            existing_match = col.get(where={"source_file": norm_path})
            old_ids = existing_match.get("ids", [])
            if old_ids:
                col.delete(ids=old_ids)
                replaced_chunks = len(old_ids)
                status = "updated"

        # Index new splits
        Chroma.from_documents(
            documents=splits,
            ids=chunk_ids,
            embedding=embeddings,
            persist_directory=DB_DIR,
            collection_name=session_name,
        )

        _vacuum_and_checkpoint()

        return {
            "status": status,
            "chunk_count": len(splits),
            "replaced_chunks": replaced_chunks,
            "session_name": session_name
        }
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