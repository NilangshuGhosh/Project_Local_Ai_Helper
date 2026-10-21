import os
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.database import (
    list_sessions,
    delete_session,
    ingest_document,
    get_retriever,
    get_database_stats,
    cleanup_orphaned_segments,
    get_session_file_summary,
)
from core.document_ops import load_and_split
from core.rag_engine import build_chat_chain
from core.health import display_health_banner, verify_system_readiness

console = Console()


def handle_list_sessions():
    sessions = list_sessions()
    stats = get_database_stats()

    if not sessions:
        console.print("[yellow]No active knowledge sessions found.[/yellow]")
        console.print(f"[dim]Vector DB storage: {stats['size_mb']} MB across {stats['file_count']} files.[/dim]\n")
        return

    table = Table(title="Saved Knowledge Sessions")
    table.add_column("Session Name", style="cyan", no_wrap=True)
    table.add_column("Files", style="magenta", justify="center")
    table.add_column("Indexed Chunks", style="green", justify="right")

    for s in sessions:
        file_count_str = str(s.get("file_count", 1))
        table.add_row(s["name"], file_count_str, str(s["count"]))

    console.print(table)
    console.print(
        f"[dim]Total: {stats['total_chunks']} chunks in {stats['session_count']} session(s) "
        f"({stats['size_mb']} MB on disk)[/dim]\n"
    )


def handle_ingest():
    file_path = questionary.path("Enter path to file (PDF, TXT, PY, CPP, JS, MD, etc.):").ask()
    if not file_path or not os.path.exists(file_path):
        console.print("[red]Invalid file path or file does not exist.[/red]\n")
        return

    default_name = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_")
    session_name = questionary.text("Name this knowledge session:", default=default_name).ask()

    if not session_name or not session_name.strip():
        console.print("[yellow]Ingestion cancelled: session name cannot be empty.[/yellow]\n")
        return

    session_name = session_name.strip()

    with console.status(f"[cyan]Analyzing, splitting, and embedding '{file_path}'...[/cyan]"):
        try:
            splits = load_and_split(file_path)
            chunk_count = ingest_document(file_path, session_name, splits)
            console.print(
                f"[bold green]✔ Successfully indexed {chunk_count} chunks with rich metadata under session '{session_name}'![/bold green]\n"
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Ingestion cancelled by user.[/yellow]\n")
        except Exception as e:
            console.print(f"[red]Error during ingestion: {e}[/red]\n")


def handle_chat():
    sessions = list_sessions()
    if not sessions:
        console.print("[yellow]No knowledge sessions available. Ingest a file first![/yellow]\n")
        return

    choices = [s["name"] for s in sessions]
    session_name = questionary.select("Select a session to chat with:", choices=choices).ask()

    if not session_name:
        return

    # Check Ollama and models before entering chat
    readiness = verify_system_readiness()
    if not readiness["ready"]:
        console.print("[bold red]Cannot start chat: System dependencies are missing.[/bold red]")
        for issue in readiness["issues"]:
            console.print(f"  • [yellow]{issue}[/yellow]")
        console.print()
        return

    # Display session file summary
    files_summary = get_session_file_summary(session_name)
    if files_summary:
        file_badges = ", ".join([f"[cyan]{f['file_name']}[/cyan] ({f['chunk_count']} chunks)" for f in files_summary])
        console.print(f"[bold dim]Documents in session:[/bold dim] {file_badges}")

    retriever = get_retriever(session_name)
    chain = build_chat_chain(retriever)

    console.print(f"[bold green]Connected to session '{session_name}'. Type 'exit' to quit.[/bold green]\n")

    while True:
        try:
            query = questionary.text("You:").ask()
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting chat session.[/yellow]\n")
            break

        if not query or query.strip().lower() in ["exit", "quit"]:
            break

        with console.status("[dim]Searching context and generating answer...[/dim]"):
            try:
                response = chain.invoke({"input": query})
                console.print(f"\n[bold magenta]AI:[/bold magenta] {response['answer']}\n")
            except Exception as e:
                console.print(f"\n[bold red]Error generating response:[/bold red] {e}\n")


def handle_delete():
    sessions = list_sessions()
    if not sessions:
        console.print("[yellow]No sessions to delete.[/yellow]\n")
        return

    choices = [s["name"] for s in sessions]
    session_name = questionary.select("Select session to delete:", choices=choices).ask()

    if not session_name:
        return

    confirm = questionary.confirm(
        f"Permanently delete '{session_name}' and physically remove its vector files from disk?"
    ).ask()

    if confirm:
        with console.status(f"[cyan]Purging '{session_name}' and cleaning physical files...[/cyan]"):
            success = delete_session(session_name)

        if success:
            console.print(
                f"[bold green]✔ Session '{session_name}' and all its physical vector files were permanently removed.[/bold green]\n"
            )
        else:
            console.print(f"[red]Failed to delete session '{session_name}'.[/red]\n")


def handle_storage_health():
    """Displays health banner, storage statistics, and allows running physical cleanup."""
    display_health_banner(console)
    stats = get_database_stats()

    info_text = (
        f"[bold]Active Sessions:[/bold] {stats['session_count']}\n"
        f"[bold]Total Chunks:[/bold] {stats['total_chunks']}\n"
        f"[bold]Disk Usage:[/bold] {stats['size_mb']} MB ({stats['file_count']} files)\n"
        f"[bold]Storage Directory:[/bold] [dim]{stats['db_dir']}[/dim]"
    )
    console.print(Panel(info_text, title="Vector DB Storage Stats", border_style="cyan"))

    action = questionary.select(
        "Storage actions:",
        choices=[
            "Run Physical Disk Cleanup (Remove orphaned files & vacuum SQLite)",
            "Back to Main Menu"
        ]
    ).ask()

    if action and "Run Physical Disk Cleanup" in action:
        with console.status("[cyan]Scanning and purging orphaned disk files...[/cyan]"):
            deleted = cleanup_orphaned_segments()
        if deleted > 0:
            console.print(f"[bold green]✔ Cleaned up {deleted} orphaned segment directory(ies) and vacuumed database.[/bold green]\n")
        else:
            console.print("[green]✔ Storage is clean! No orphaned directories found. SQLite vacuumed.[/green]\n")