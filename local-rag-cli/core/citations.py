import os
import sys
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def truncate_snippet(text: str, max_chars: int = 140) -> str:
    """Cleans and truncates a text excerpt for preview display."""
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def render_sources_panel(console: Console, docs: List[Document]):
    """
    Renders a compact, verified sources panel below the AI response,
    mapping each [Source N] tag to its file, page, paragraph, and excerpt.
    """
    if not docs:
        return

    table = Table(
        title="[bold green]Verified Sources Used[/bold green]",
        box=None,
        padding=(0, 1),
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Tag", style="bold yellow", width=10, no_wrap=True)
    table.add_column("Document & Location", style="cyan", width=32)
    table.add_column("Excerpt Preview", style="dim", ratio=1)

    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        raw_source = meta.get("source_file") or meta.get("source", "")
        file_name = meta.get("file_name") or (os.path.basename(raw_source) if raw_source else "Unknown Document")
        page = meta.get("page")
        para = meta.get("paragraph_idx")
        file_type = meta.get("file_type", "")

        # Format location badge (Windows console friendly)
        if page is not None and file_type == "pdf":
            location = f"{file_name} [bold](Page {page}, Para {para})[/bold]"
        elif para is not None:
            location = f"{file_name} [bold](Para {para})[/bold]"
        else:
            location = file_name

        snippet = truncate_snippet(doc.page_content, max_chars=120)
        table.add_row(f"[Source {idx}]", location, f'"{snippet}"')

    console.print(table)
    console.print(
        "[dim]Actions: Type [bold cyan]'/open <N>'[/bold cyan] to open file in Windows, "
        "[bold cyan]'/source <N>'[/bold cyan] for full excerpt, or ask your next question.[/dim]\n"
    )


def open_source_file_in_windows(doc: Document) -> (bool, str):
    """
    Launches the source file using Windows default viewer (e.g. Adobe Acrobat / Edge for PDF,
    Notepad for TXT, IDE for code).
    """
    meta = doc.metadata or {}
    source_path = meta.get("source_file") or meta.get("source", "")

    if not source_path or not os.path.exists(source_path):
        return False, f"Source file not found on disk: {source_path}"

    try:
        # On Windows, os.startfile opens the file with its associated default program
        os.startfile(os.path.normpath(source_path))
        return True, f"Opened '{os.path.basename(source_path)}' in default application."
    except Exception as e:
        return False, f"Could not launch file: {e}"


def display_full_source_detail(console: Console, doc: Document, index: int):
    """
    Displays the full, uncropped content of a specific retrieved source chunk
    along with its complete metadata audit.
    """
    meta = doc.metadata or {}
    raw_source = meta.get("source_file") or meta.get("source", "")
    file_name = meta.get("file_name") or (os.path.basename(raw_source) if raw_source else "Unknown Document")
    page_str = f"Page {meta.get('page')} | " if meta.get("page") else ""
    para_str = f"Paragraph {meta.get('paragraph_idx')} | " if meta.get("paragraph_idx") else ""
    chars = len(doc.page_content)

    title = f"[bold green]Source Chunk [{index}]: {file_name}[/bold green]"
    sub_info = f"[dim]{page_str}{para_str}{chars} chars | Path: {raw_source}[/dim]\n"

    content = f"{sub_info}\n{doc.page_content}"
    console.print(Panel(content, title=title, border_style="cyan", expand=False))
