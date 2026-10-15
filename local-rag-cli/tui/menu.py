import os
import questionary
from rich.console import Console
from rich.table import Table

from core.database import list_sessions, delete_session, ingest_document, get_retriever
from core.document_ops import load_and_split
from core.rag_engine import build_chat_chain

console = Console()

def handle_list_sessions():
    sessions = list_sessions()
    if not sessions:
        console.print("[yellow]No active knowledge sessions found.[/yellow]\n")
        return

    table = Table(title="Saved Knowledge Sessions")
    table.add_column("Session Name", style="cyan")
    table.add_column("Chunks", style="green")

    for s in sessions:
        table.add_row(s["name"], str(s["count"]))

    console.print(table)
    print()

def handle_ingest():
    file_path = questionary.path("Enter path to file (PDF, TXT, CPP):").ask()
    if not file_path or not os.path.exists(file_path):
        console.print("[red]Invalid file path.[/red]\n")
        return

    default_name = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_")
    session_name = questionary.text("Name this knowledge session:", default=default_name).ask()

    if not session_name:
        return

    with console.status(f"[cyan]Reading and indexing '{file_path}'...[/cyan]"):
        try:
            splits = load_and_split(file_path)
            ingest_document(file_path, session_name, splits)
            console.print(f"[green]Successfully indexed {len(splits)} chunks under '{session_name}'![/green]\n")
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

    retriever = get_retriever(session_name)
    chain = build_chat_chain(retriever)

    console.print(f"[bold green]Connected to session '{session_name}'. Type 'exit' to quit.[/bold green]\n")

    while True:
        query = questionary.text("You:").ask()
        if not query or query.strip().lower() in ["exit", "quit"]:
            break

        with console.status("[dim]Searching and thinking...[/dim]"):
            response = chain.invoke({"input": query})
            
        console.print(f"\n[bold magenta]AI:[/bold magenta] {response['answer']}\n")

def handle_delete():
    sessions = list_sessions()
    if not sessions:
        console.print("[yellow]No sessions to delete.[/yellow]\n")
        return

    choices = [s["name"] for s in sessions]
    session_name = questionary.select("Select session to delete:", choices=choices).ask()

    if not session_name:
        return

    confirm = questionary.confirm(f"Permanently delete '{session_name}'?").ask()
    if confirm:
        if delete_session(session_name):
            console.print(f"[green]Session '{session_name}' deleted successfully.[/green]\n")
        else:
            console.print(f"[red]Failed to delete session.[/red]\n")