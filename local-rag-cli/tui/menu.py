import os
import sys
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.database import (
    list_sessions,
    delete_session,
    delete_file_from_session,
    ingest_document,
    get_retriever,
    get_database_stats,
    cleanup_orphaned_segments,
    get_session_file_summary,
)
from core.document_ops import load_and_split
from core.rag_engine import stream_rag_response, build_chat_chain
from core.health import display_health_banner, verify_system_readiness
from core.citations import (
    render_sources_panel,
    open_source_file_in_windows,
    display_full_source_detail,
)
from core.memory import memory_manager

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


def handle_ingest(target_session: str = None):
    """
    Ingests a new document, allowing the user to create a new session
    or add to an existing multi-document session.
    """
    file_path = questionary.path("Enter path to file (PDF, TXT, PY, CPP, JS, MD, etc.):").ask()
    if not file_path or not os.path.exists(file_path):
        console.print("[red]Invalid file path or file does not exist.[/red]\n")
        return

    file_name = os.path.basename(file_path)

    # Determine session name
    if target_session:
        session_name = target_session
    else:
        existing_sessions = [s["name"] for s in list_sessions()]
        if existing_sessions:
            dest_choice = questionary.select(
                "Where would you like to ingest this document?",
                choices=[
                    "Create a New Session",
                    "Add to an Existing Session"
                ]
            ).ask()

            if dest_choice == "Add to an Existing Session":
                session_name = questionary.select(
                    "Select target session:",
                    choices=existing_sessions
                ).ask()
                if not session_name:
                    return
            else:
                default_name = os.path.splitext(file_name)[0].replace(" ", "_")
                session_name = questionary.text("Name this knowledge session:", default=default_name).ask()
        else:
            default_name = os.path.splitext(file_name)[0].replace(" ", "_")
            session_name = questionary.text("Name this knowledge session:", default=default_name).ask()

    if not session_name or not session_name.strip():
        console.print("[yellow]Ingestion cancelled: session name cannot be empty.[/yellow]\n")
        return

    session_name = session_name.strip()

    with console.status(f"[cyan]Analyzing, splitting, and embedding '{file_name}' into '{session_name}'...[/cyan]"):
        try:
            splits = load_and_split(file_path)
            res = ingest_document(file_path, session_name, splits)
            
            if res["status"] == "updated":
                console.print(
                    f"[bold green]✔ Updated '{file_name}' in session '{session_name}'![/bold green] "
                    f"[dim](Replaced {res['replaced_chunks']} stale chunks with {res['chunk_count']} new chunks)[/dim]\n"
                )
            else:
                console.print(
                    f"[bold green]✔ Added '{file_name}' to session '{session_name}' ({res['chunk_count']} chunks indexed)![/bold green]\n"
                )
        except KeyboardInterrupt:
            console.print("\n[yellow]Ingestion cancelled by user.[/yellow]\n")
        except Exception as e:
            console.print(f"[red]Error during ingestion: {e}[/red]\n")


def handle_manage_session_docs():
    """
    Interactive manager to inspect, add, or selectively remove documents
    within a specific session.
    """
    sessions = list_sessions()
    if not sessions:
        console.print("[yellow]No active knowledge sessions found.[/yellow]\n")
        return

    choices = [s["name"] for s in sessions] + ["< Back"]
    session_name = questionary.select("Select a session to inspect/manage:", choices=choices).ask()

    if not session_name or session_name == "< Back":
        return

    files = get_session_file_summary(session_name)
    if not files:
        console.print(f"[yellow]No file metadata records found in session '{session_name}'.[/yellow]\n")
        return

    table = Table(title=f"Documents in Session: {session_name}")
    table.add_column("Document Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta", justify="center")
    table.add_column("Pages", style="blue", justify="center")
    table.add_column("Chunks", style="green", justify="right")
    table.add_column("Path", style="dim")

    for f in files:
        table.add_row(
            f["display_name"],
            f["file_type"],
            str(f.get("total_pages", 1)),
            str(f["chunk_count"]),
            f["source_file"]
        )

    console.print(table)
    console.print()

    sub_action = questionary.select(
        f"Manage documents in '{session_name}':",
        choices=[
            "Remove a document from this session",
            "Add another document to this session",
            "< Back to Main Menu"
        ]
    ).ask()

    if sub_action == "Remove a document from this session":
        file_choices = [
            questionary.Choice(
                title=f"{f['display_name']} ({f['chunk_count']} chunks)",
                value=f["source_file"]
            )
            for f in files
        ] + [questionary.Choice(title="< Cancel", value=None)]

        target_file = questionary.select("Select file to remove:", choices=file_choices).ask()
        if not target_file:
            return

        confirm = questionary.confirm(f"Remove '{os.path.basename(target_file)}' from session '{session_name}'?").ask()
        if confirm:
            with console.status(f"[cyan]Removing document and cleaning chunks...[/cyan]"):
                del_res = delete_file_from_session(session_name, target_file, by_path=True)

            if del_res["success"]:
                if del_res["session_deleted"]:
                    console.print(
                        f"[bold green]✔ Removed '{os.path.basename(target_file)}'.[/bold green] "
                        f"[yellow]Session '{session_name}' had 0 remaining files and was purged.[/yellow]\n"
                    )
                else:
                    console.print(
                        f"[bold green]✔ Successfully removed '{os.path.basename(target_file)}'![/bold green] "
                        f"[dim]Deleted {del_res['deleted_chunks']} chunks. {del_res['remaining_chunks']} chunks remaining in session.[/dim]\n"
                    )
            else:
                console.print(f"[red]Failed to remove file: {del_res['error']}[/red]\n")

    elif sub_action == "Add another document to this session":
        handle_ingest(target_session=session_name)


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
        file_badges = ", ".join([f"[cyan]{f['display_name']}[/cyan] ({f['chunk_count']} chunks)" for f in files_summary])
        console.print(f"[bold dim]Documents in session:[/bold dim] {file_badges}")

    retriever = get_retriever(session_name)

    console.print(f"[bold green]Connected to session '{session_name}'.[/bold green] [dim](Type 'exit' to quit, '/help' for commands)[/dim]\n")

    last_retrieved_docs = []

    while True:
        try:
            query = questionary.text("You:").ask()
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting chat session.[/yellow]\n")
            break

        if not query or query.strip().lower() in ["exit", "quit"]:
            break

        cleaned_query = query.strip()

        # Handle interactive slash commands
        if cleaned_query.lower() == "/help":
            console.print(
                "\n[bold cyan]Interactive Chat Commands:[/bold cyan]\n"
                "  • [bold]/sources[/bold]     : Re-display the verified sources table for the last answer\n"
                "  • [bold]/source <N>[/bold]  : View the complete uncropped text and metadata of Source N\n"
                "  • [bold]/open <N>[/bold]    : Open Source N's file in your default Windows viewer\n"
                "  • [bold]/history[/bold]     : View previous conversation turns in this session\n"
                "  • [bold]/clear[/bold]       : Reset conversation memory for this session (start fresh topic)\n"
                "  • [bold]exit[/bold] or [bold]quit[/bold]: Return to the main menu\n"
            )
            continue

        if cleaned_query.lower() == "/clear":
            cleared = memory_manager.clear(session_name)
            if cleared:
                console.print("[bold green]✔ Conversation memory cleared for this session.[/bold green]\n")
            else:
                console.print("[dim]Conversation memory is already empty.[/dim]\n")
            continue

        if cleaned_query.lower() == "/history":
            turns = memory_manager.get_turns(session_name)
            if not turns:
                console.print("[yellow]No conversation history in memory yet.[/yellow]\n")
            else:
                console.print(f"\n[bold cyan]Conversation History for '{session_name}' ({len(turns)} turns):[/bold cyan]")
                for i, t in enumerate(turns, 1):
                    console.print(f"[bold dim]Turn {i}:[/bold dim]")
                    console.print(f"  [bold]User:[/bold] {t.user_query}")
                    console.print(f"  [bold magenta]AI:[/bold magenta] {t.assistant_response}")
                console.print()
            continue

        if cleaned_query.lower() == "/sources":
            if last_retrieved_docs:
                render_sources_panel(console, last_retrieved_docs)
            else:
                console.print("[yellow]No sources available yet. Ask a question first![/yellow]\n")
            continue

        if cleaned_query.lower().startswith("/source"):
            parts = cleaned_query.split()
            if len(parts) < 2 or not parts[1].isdigit():
                console.print("[yellow]Usage: /source <number> (e.g. /source 1)[/yellow]\n")
            else:
                idx = int(parts[1])
                if 1 <= idx <= len(last_retrieved_docs):
                    display_full_source_detail(console, last_retrieved_docs[idx - 1], idx)
                else:
                    console.print(f"[red]Source [{idx}] out of range. Last answer cited {len(last_retrieved_docs)} source(s).[/red]\n")
            continue

        if cleaned_query.lower().startswith("/open"):
            parts = cleaned_query.split()
            if len(parts) < 2 or not parts[1].isdigit():
                console.print("[yellow]Usage: /open <number> (e.g. /open 1)[/yellow]\n")
            else:
                idx = int(parts[1])
                if 1 <= idx <= len(last_retrieved_docs):
                    target_doc = last_retrieved_docs[idx - 1]
                    success, msg = open_source_file_in_windows(target_doc)
                    if success:
                        console.print(f"[bold green]✔ {msg}[/bold green]\n")
                    else:
                        console.print(f"[red]{msg}[/red]\n")
                else:
                    console.print(f"[red]Source [{idx}] out of range. Last answer cited {len(last_retrieved_docs)} source(s).[/red]\n")
            continue

        # Show animated spinner during document retrieval & prompt prefill
        stream = stream_rag_response(retriever, query, session_name=session_name)
        first_token = None
        context_docs = []
        error_msg = None
        full_answer = ""

        try:
            with console.status("[dim cyan]Searching documents & preparing response...[/dim cyan]", spinner="dots"):
                for event in stream:
                    ev_type = event.get("type")
                    if ev_type == "context":
                        context_docs = event.get("docs", [])
                    elif ev_type == "token":
                        first_token = event.get("token")
                        break
                    elif ev_type == "error":
                        error_msg = event.get("error")
                        break

            if error_msg:
                console.print(f"\n[bold red]Error:[/bold red] {error_msg}\n")
                continue

            # Spinner has cleared; stream tokens live to terminal
            console.print("\n[bold magenta]AI:[/bold magenta] ", end="")
            if first_token:
                full_answer += first_token
                sys.stdout.write(first_token)
                sys.stdout.flush()

            for event in stream:
                ev_type = event.get("type")
                if ev_type == "token":
                    full_answer += event["token"]
                    sys.stdout.write(event["token"])
                    sys.stdout.flush()
                elif ev_type == "error":
                    console.print(f"\n[bold red]Error:[/bold red] {event['error']}")
                elif ev_type == "done":
                    ttft = event.get("ttft_s", 0)
                    total_s = event.get("elapsed_s", 0)
                    tokens = event.get("total_tokens", 0)
                    gen_time = max(total_s - ttft, 0.05)
                    gen_tok_per_sec = round(tokens / gen_time, 1)
                    console.print(f"\n[dim]({tokens} tokens • TTFT: {ttft}s • Generation: {gen_tok_per_sec} tok/s • Total: {total_s}s)[/dim]\n")

            # Record turn in conversation memory if answer was generated
            if full_answer.strip():
                memory_manager.add_turn(session_name, query, full_answer)

            # Render verified sources panel below response
            if context_docs:
                last_retrieved_docs = context_docs
                render_sources_panel(console, context_docs)

        except KeyboardInterrupt:
            console.print("\n[yellow][Generation stopped by user][/yellow]\n")
        except Exception as e:
            console.print(f"\n[bold red]Streaming error:[/bold red] {e}\n")


def handle_delete():
    sessions = list_sessions()
    if not sessions:
        console.print("[yellow]No sessions to delete.[/yellow]\n")
        return

    del_mode = questionary.select(
        "What would you like to delete?",
        choices=[
            "Delete an Entire Session (and all its files)",
            "Remove a Specific Document from a Session",
            "< Cancel"
        ]
    ).ask()

    if del_mode == "< Cancel" or not del_mode:
        return

    if del_mode == "Remove a Specific Document from a Session":
        handle_manage_session_docs()
        return

    choices = [s["name"] for s in sessions]
    session_name = questionary.select("Select session to delete permanently:", choices=choices).ask()

    if not session_name:
        return

    confirm = questionary.confirm(
        f"Permanently delete session '{session_name}' and physically remove all its vector files from disk?"
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