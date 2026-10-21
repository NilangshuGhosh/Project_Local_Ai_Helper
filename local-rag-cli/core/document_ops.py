import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from core.config import CHUNK_SIZE, CHUNK_OVERLAP, CODE_CHUNK_SIZE, CODE_CHUNK_OVERLAP

# Mapping of file extensions to LangChain Language splitters
CODE_EXT_MAP = {
    ".py": Language.PYTHON,
    ".cpp": Language.CPP,
    ".hpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".c": Language.C,
    ".h": Language.C,
    ".js": Language.JS,
    ".mjs": Language.JS,
    ".cjs": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".jsx": Language.JS,
    ".java": Language.JAVA,
    ".rs": Language.RUST,
    ".go": Language.GO,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
}


def _split_pdf_page_by_page(file_path: str, norm_path: str, file_name: str) -> List[Document]:
    """
    Loads a PDF and splits each page independently so that chunks NEVER bleed across
    page boundaries. Enriches each chunk with 1-indexed page, paragraph index, and file metadata.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    enriched_chunks: List[Document] = []
    global_chunk_idx = 1

    for page_doc in pages:
        # PyPDFLoader pages are 0-indexed; convert to 1-indexed for human readability
        raw_page_num = page_doc.metadata.get("page", 0)
        page_num = raw_page_num + 1 if isinstance(raw_page_num, int) else 1

        page_content = page_doc.page_content or ""
        if not page_content.strip():
            continue

        page_splits = text_splitter.split_text(page_content)

        for p_idx, chunk_text in enumerate(page_splits, start=1):
            cleaned = chunk_text.strip()
            if not cleaned:
                continue

            metadata = {
                "source_file": norm_path,
                "file_name": file_name,
                "file_type": "pdf",
                "page": page_num,
                "chunk_index": global_chunk_idx,
                "paragraph_idx": p_idx,
                "char_count": len(cleaned),
            }
            enriched_chunks.append(Document(page_content=cleaned, metadata=metadata))
            global_chunk_idx += 1

    return enriched_chunks


def _split_code_file(file_path: str, norm_path: str, file_name: str, language: Language) -> List[Document]:
    """
    Splits code files using language-aware AST-like boundary splitters
    (preserving class/function definitions and indentation levels).
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=CODE_CHUNK_SIZE,
        chunk_overlap=CODE_CHUNK_OVERLAP,
    )

    raw_chunks = splitter.split_documents(docs)
    enriched_chunks: List[Document] = []

    for idx, chunk in enumerate(raw_chunks, start=1):
        cleaned = chunk.page_content.strip()
        if not cleaned:
            continue

        metadata = {
            "source_file": norm_path,
            "file_name": file_name,
            "file_type": "code",
            "language": language.value,
            "page": 1,
            "chunk_index": idx,
            "paragraph_idx": idx,
            "char_count": len(cleaned),
        }
        enriched_chunks.append(Document(page_content=cleaned, metadata=metadata))

    return enriched_chunks


def _split_text_file(file_path: str, norm_path: str, file_name: str) -> List[Document]:
    """
    Splits plain text, markdown, or config files prioritizing paragraph boundaries.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    raw_chunks = splitter.split_documents(docs)
    enriched_chunks: List[Document] = []

    for idx, chunk in enumerate(raw_chunks, start=1):
        cleaned = chunk.page_content.strip()
        if not cleaned:
            continue

        metadata = {
            "source_file": norm_path,
            "file_name": file_name,
            "file_type": "text",
            "page": 1,
            "chunk_index": idx,
            "paragraph_idx": idx,
            "char_count": len(cleaned),
        }
        enriched_chunks.append(Document(page_content=cleaned, metadata=metadata))

    return enriched_chunks


def load_and_split(file_path: str) -> List[Document]:
    """
    Main entry point for document loading and precision chunking.
    
    Inspects the file extension, applies the appropriate content-aware splitter,
    and returns Document objects enriched with page numbers, paragraph positions,
    and source metadata for citations and CRUD operations.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise ValueError(f"File is empty: {file_path}")

    norm_path = os.path.abspath(file_path).replace("\\", "/")
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        chunks = _split_pdf_page_by_page(file_path, norm_path, file_name)
    elif ext in CODE_EXT_MAP:
        chunks = _split_code_file(file_path, norm_path, file_name, CODE_EXT_MAP[ext])
    else:
        chunks = _split_text_file(file_path, norm_path, file_name)

    if not chunks:
        raise ValueError(f"No valid text content could be extracted from: {file_path}")

    return chunks