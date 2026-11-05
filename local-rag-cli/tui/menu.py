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
import datetime
from core.memory import memory_manager
from core.config import SUPPORTED_MODELS, get_active_model, set_active_model
from core.history_db import (
    save_chat_turn,
    get_recent_session_turns,
    get_all_session_turns,
    get_session_message_count,
    delete_session_chat_history,
    search_chat_history,
)
from core.hardware import get_system_hardware_info, evaluate_model_safety

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


def pick_file_with_native_dialog() -> str:
    """
    Opens the native OS File Explorer dialog to allow the user to browse and select
    a document visually without typing file paths by hand.
    Falls back gracefully to manual text input if the GUI environment is unavailable.
    """
    selected_path = ""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        # Bring dialog to the front of Windows terminal
        root.attributes("-topmost", True)
        root.focus_force()

        filetypes = [
            ("All Supported Documents", "*.pdf;*.txt;*.md;*.py;*.cpp;*.c;*.h;*.hpp;*.js;*.ts;*.java;*.rs;*.go;*.html;*.json;*.csv"),
            ("PDF Documents (*.pdf)", "*.pdf"),
            ("Text & Markdown (*.txt, *.md)", "*.txt;*.md"),
            ("Source Code (*.py, *.js, *.cpp, ...)", "*.py;*.js;*.ts;*.cpp;*.c;*.java;*.rs;*.go"),
            ("All Files (*.*)", "*.*")
        ]

        selected_path = filedialog.askopenfilename(
            title="Select Document to Ingest into Local AI Helper",
            filetypes=filetypes
        )
        root.destroy()
    except Exception:
        selected_path = ""

    return selected_path


def handle_ingest(target_session: str = None):
    """
    Ingests a new document, allowing the user to select via native Windows File Explorer
    or manual path input, and create a new session or add to an existing multi-document session.
    """
    # Offer user choice between Native Explorer Dialog and Manual Typing
    picker_mode = questionary.select(
        "How would you like to select your document?",
        choices=[
            "Browse via Windows File Explorer (Recommended)",
            "Type / Paste File Path Manually",
            "< Cancel"
        ]
    ).ask()

    if not picker_mode or picker_mode == "< Cancel":
        return

    file_path = ""
    if picker_mode == "Browse via Windows File Explorer (Recommended)":
        console.print("[dim cyan]Opening Windows File Explorer dialog...[/dim cyan]")
        file_path = pick_file_with_native_dialog()
        if not file_path:
            console.print("[yellow]No file selected in File Explorer. Ingestion cancelled.[/yellow]\n")
            return
        console.print(f"[bold green]✔ Selected:[/bold green] [cyan]{file_path}[/cyan]")
    else:
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
        ] + ["< Cancel"]

        target_file = questionary.select("Select file to remove:", choices=file_choices).ask()
        if not target_file or target_file == "< Cancel":
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


def handle_chat(target_session: str = None):
    sessions = list_sessions()
    if not sessions:
        console.print("[yellow]No knowledge sessions available. Ingest a file first![/yellow]\n")
        return

    if target_session:
        session_name = target_session
    else:
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

    # -------------------------------------------------------------------------
    # SESSION RESUMPTION: Load and display previous chat history if available
    # -------------------------------------------------------------------------
    recent_saved = get_recent_session_turns(session_name, limit=3)
    total_saved = get_session_message_count(session_name)

    if recent_saved:
        console.print(f"\n[bold cyan]── Previous Conversation History ({total_saved} total messages on disk) ──[/bold cyan]")
        for row in recent_saved:
            time_str = row.get("timestamp", "")
            console.print(f"[bold dim][{time_str}][/bold dim]")
            console.print(f"  [bold]User:[/bold] {row['user_query']}")
            console.print(f"  [bold magenta]AI:[/bold magenta] {row['ai_response']}")
            # Pre-load into in-memory sliding buffer so follow-up pronouns work immediately
            memory_manager.add_turn(session_name, row['user_query'], row['ai_response'])
        console.print("[bold cyan]──────────────────────────────────────────────────────────────────[/bold cyan]\n")

    console.print(f"[bold green]Connected to session '{session_name}'.[/bold green] [dim](Type 'exit' to quit, '/help' for commands)[/dim]\n")

    last_retrieved_docs = []
    is_incognito = False

    while True:
        try:
            if is_incognito:
                prompt_prefix = "🕶  You (Incognito):"
                incog_style = questionary.Style([
                    ("qmark", "fg:#e5c07b bold"),
                    ("question", "fg:#e5c07b bold"),
                ])
                query = questionary.text(prompt_prefix, style=incog_style).ask()
            else:
                prompt_prefix = "You:"
                query = questionary.text(prompt_prefix).ask()
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
                "  • [bold]/model[/bold]       : View or switch active LLM (e.g. Qwen 2.5 vs Phi-3)\n"
                "  • [bold]/incognito[/bold]   : Toggle Incognito Mode ON/OFF (RAM-only, zero disk logging)\n"
                "  • [bold]/history[/bold]     : View bi-directional paginated transcript of past Q&A\n"
                "  • [bold]/search <term>[/bold]: Search past interactions in this session or all sessions\n"
                "  • [bold]/export[/bold]      : Export conversation transcript with sources to Markdown file\n"
                "  • [bold]/sources[/bold]     : Re-display the verified sources table for the last answer\n"
                "  • [bold]/source <N>[/bold]  : View the complete uncropped text and metadata of Source N\n"
                "  • [bold]/open <N>[/bold]    : Open Source N's file in your default Windows viewer\n"
                "  • [bold]/clear[/bold]       : Reset conversation memory (RAM & disk history for this session)\n"
                "  • [bold]exit[/bold] or [bold]quit[/bold]: Return to the main menu\n"
            )
            continue

        if cleaned_query.lower() == "/incognito":
            is_incognito = not is_incognito
            if is_incognito:
                console.print("\n[bold yellow]🕶  Incognito Mode is now ON.[/bold yellow] [dim]Conversations are kept strictly in RAM and will NOT be saved to disk.[/dim]\n")
            else:
                console.print("\n[bold green]🕶  Incognito Mode is now OFF.[/bold green] [dim]Persistent disk logging resumed.[/dim]\n")
            continue

        if cleaned_query.lower() == "/export":
            export_turns = get_all_session_turns(session_name)
            # If nothing in SQLite (e.g. incognito session), check memory manager
            if not export_turns:
                ram_turns = memory_manager.get_turns(session_name)
                export_turns = [
                    {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "user_query": t.user_query,
                        "ai_response": t.assistant_response,
                        "sources_summary": "Active Session RAM",
                        "model_used": get_active_model()
                    }
                    for t in ram_turns
                ]

            if not export_turns:
                console.print("[yellow]No conversation messages to export yet.[/yellow]\n")
                continue

            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"{session_name}_transcript_{timestamp_str}.md"
            default_export_path = os.path.join(desktop_path, default_filename)

            export_loc_choice = questionary.select(
                "Export destination:",
                choices=[
                    f"Save to Windows Desktop ({default_filename})",
                    "Choose Another Folder via File Explorer",
                    "< Cancel"
                ]
            ).ask()

            if not export_loc_choice or export_loc_choice == "< Cancel":
                continue

            final_save_path = default_export_path
            if "Choose Another Folder" in export_loc_choice:
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    chosen_dir = filedialog.askdirectory(title="Select Folder to Save Transcript")
                    root.destroy()
                    if chosen_dir:
                        final_save_path = os.path.join(chosen_dir, default_filename)
                except Exception:
                    final_save_path = default_export_path

            # Format Markdown document
            md_lines = [
                f"# Local AI Helper - Chat Transcript: {session_name}",
                f"*Export Date:* {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"*Total Interactions:* {len(export_turns)}",
                "",
                "---",
                ""
            ]

            for i, turn in enumerate(export_turns, 1):
                t_stamp = turn.get("timestamp", "")
                m_used = turn.get("model_used") or get_active_model()
                srcs = turn.get("sources_summary", "")
                md_lines.append(f"### Turn {i} ({t_stamp}) - Model: `{m_used}`")
                md_lines.append(f"**User:** {turn['user_query']}\n")
                md_lines.append(f"**AI:** {turn['ai_response']}\n")
                if srcs:
                    md_lines.append(f"> *Sources:* {srcs}\n")
                md_lines.append("")

            try:
                with open(final_save_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(md_lines))
                console.print(f"[bold green]✔ Successfully exported transcript to:[/bold green] [cyan]{final_save_path}[/cyan]\n")
            except Exception as e:
                console.print(f"[red]Failed to export transcript: {e}[/red]\n")
            continue

        if cleaned_query.lower() == "/model":
            handle_switch_model()
            continue

        if cleaned_query.lower() == "/clear":
            cleared_ram = memory_manager.clear(session_name)
            cleared_db = delete_session_chat_history(session_name)
            console.print(f"[bold green]✔ Conversation memory cleared for '{session_name}'.[/bold green] [dim](Purged RAM buffer & {cleared_db} disk records)[/dim]\n")
            continue

        if cleaned_query.lower() == "/history":
            all_turns = get_all_session_turns(session_name)
            if not all_turns:
                ram_turns = memory_manager.get_turns(session_name)
                if not ram_turns:
                    console.print("[yellow]No conversation history in memory or on disk yet.[/yellow]\n")
                    continue
                all_turns = [
                    {"timestamp": "RAM buffer", "user_query": t.user_query, "ai_response": t.assistant_response}
                    for t in ram_turns
                ]

            total_turns = len(all_turns)
            page_size = 5
            total_pages = max(1, (total_turns + page_size - 1) // page_size)
            current_page = 1

            while True:
                start_idx = (current_page - 1) * page_size
                end_idx = min(start_idx + page_size, total_turns)
                page_slice = all_turns[start_idx:end_idx]

                console.print(f"\n[bold cyan]Conversation History for '{session_name}' • Page {current_page} of {total_pages} (Turns {start_idx + 1} to {end_idx} of {total_turns}):[/bold cyan]")
                for i, t in enumerate(page_slice, start_idx + 1):
                    t_time = t.get("timestamp", "")
                    console.print(f"[bold dim]── Turn {i} [{t_time}] ──[/bold dim]")
                    console.print(f"  [bold]User:[/bold] {t['user_query']}")
                    console.print(f"  [bold magenta]AI:[/bold magenta] {t['ai_response']}\n")

                if total_pages == 1:
                    break

                nav_choices = []
                if current_page < total_pages:
                    nav_choices.append(questionary.Choice(f"▶ Next Page ({current_page + 1}/{total_pages})", value="next"))
                if current_page > 1:
                    nav_choices.append(questionary.Choice(f"◀ Previous Page ({current_page - 1}/{total_pages})", value="prev"))
                if total_pages > 1:
                    nav_choices.append(questionary.Choice("🔢 Jump to Specific Page", value="jump"))
                nav_choices.append(questionary.Choice("✖ Close History Viewer", value="exit"))

                action = questionary.select(
                    f"Navigate History (Page {current_page}/{total_pages} • Use ↑/↓ to choose):",
                    choices=nav_choices
                ).ask()

                if not action or action == "exit":
                    break
                elif action == "next":
                    current_page += 1
                elif action == "prev":
                    current_page -= 1
                elif action == "jump":
                    page_str = questionary.text(f"Enter page number (1 to {total_pages}):").ask()
                    if page_str and page_str.isdigit():
                        p = int(page_str)
                        if 1 <= p <= total_pages:
                            current_page = p
                        else:
                            console.print(f"[yellow]Page {p} is out of range.[/yellow]")

            console.print()
            continue

        if cleaned_query.lower().startswith("/search"):
            # 1. First prompt user for search scope: Local session vs Global
            scope_choice = questionary.select(
                "Select Search Scope:",
                choices=[
                    f"Current Session Only ('{session_name}')",
                    "Global Search (All Sessions Across Knowledge Vault)",
                    "< Cancel"
                ]
            ).ask()

            if not scope_choice or scope_choice == "< Cancel":
                continue

            target_scope = session_name if "Current Session" in scope_choice else None

            # 2. Extract or prompt for search query
            parts = cleaned_query.split(maxsplit=1)
            search_term = parts[1].strip() if len(parts) > 1 else ""
            if not search_term:
                search_term = questionary.text("Enter search query or keyword:").ask()

            if not search_term or not search_term.strip():
                continue

            with console.status(f"[cyan]Searching interactions for '{search_term}'...[/cyan]"):
                results = search_chat_history(search_term, session_name=target_scope, limit=30)

            if not results:
                console.print(f"[yellow]No interactions found matching '{search_term}'.[/yellow]\n")
                continue

            # 3. Interactive selectable list to prevent terminal screen flooding
            choices = []
            for r in results:
                s_name = r.get("session_name", "")
                t_stamp = r.get("timestamp", "")
                short_q = r["user_query"].replace("\n", " ").strip()
                if len(short_q) > 60:
                    short_q = short_q[:57] + "..."
                display_title = f"[{s_name}] \"{short_q}\" — {t_stamp}"
                choices.append(questionary.Choice(title=display_title, value=r))

            choices.append(questionary.Choice(title="< Back to Chat", value=None))

            selected_match = questionary.select(
                f"Found {len(results)} match(es) for '{search_term}' (Select to view, ↑/↓ to scroll):",
                choices=choices
            ).ask()

            if not selected_match or not isinstance(selected_match, dict):
                continue

            # Render selected match in a clean card
            hit_session = selected_match.get("session_name", "")
            t_stamp = selected_match.get("timestamp", "")
            m_used = selected_match.get("model_used") or "Unknown"
            srcs = selected_match.get("sources_summary") or "None"

            panel_content = (
                f"[bold cyan]Session:[/bold cyan] {hit_session} | [bold cyan]Timestamp:[/bold cyan] {t_stamp}\n"
                f"[bold cyan]Model:[/bold cyan] `{m_used}` | [bold cyan]Sources:[/bold cyan] {srcs}\n\n"
                f"[bold]User:[/bold] {selected_match['user_query']}\n\n"
                f"[bold magenta]AI:[/bold magenta] {selected_match['ai_response']}"
            )
            console.print(Panel(panel_content, title=f"Match Details: {hit_session}", border_style="cyan"))

            # If match came from another session (via Global Search), offer 1-click jump!
            if hit_session != session_name:
                jump_confirm = questionary.confirm(
                    f"This interaction is from session '{hit_session}'. Jump to that session now?",
                    default=False
                ).ask()
                if jump_confirm:
                    handle_chat(target_session=hit_session)
                    break
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

            # -----------------------------------------------------------------
            # RECORD TURN: Memory Buffer + SQLite Persistent Storage
            # -----------------------------------------------------------------
            if full_answer.strip():
                # 1. Always record in RAM buffer for active pronoun resolution
                memory_manager.add_turn(session_name, query, full_answer)

                # 2. Persist to SQLite ONLY if Incognito Mode is OFF
                if not is_incognito:
                    src_summaries = []
                    for d in (context_docs or []):
                        m = d.metadata or {}
                        fn = m.get("file_name") or os.path.basename(m.get("source_file", "Doc"))
                        pg = m.get("page")
                        src_summaries.append(f"{fn} (p.{pg})" if pg else fn)
                    
                    sources_str = ", ".join(dict.fromkeys(src_summaries))
                    save_chat_turn(
                        session_name=session_name,
                        user_query=query,
                        ai_response=full_answer,
                        sources_summary=sources_str,
                        model_used=get_active_model()
                    )

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


def handle_switch_model():
    """Interactive selector to switch the active LLM between installed models at runtime with hardware safety evaluation."""
    active = get_active_model()
    hw = get_system_hardware_info()
    
    choices = []
    for tag, info in SUPPORTED_MODELS.items():
        is_active = (tag == active)
        prefix = "★ [ACTIVE] " if is_active else "  "
        safety = evaluate_model_safety(tag, hw)
        badge = safety["menu_badge"]
        title = f"{prefix}{info['name']} — {info['description']} {badge}"
        choices.append(questionary.Choice(title=title, value=tag))

    choices.append("< Cancel (Keep current)")

    selected = questionary.select(
        f"Select Active LLM Model (Currently: {active}):",
        choices=choices
    ).ask()

    if not selected or selected == "< Cancel (Keep current)" or selected == active:
        if selected == active:
            console.print(f"[dim]Model '{active}' is already active.[/dim]\n")
        return

    safety = evaluate_model_safety(selected, hw)
    if safety["safety_level"] in ["CAUTION", "CRITICAL"]:
        console.print(f"\n[bold yellow]⚠️ Hardware Notice for '{selected}':[/bold yellow]")
        console.print(f"[yellow]{safety['warning']}[/yellow]\n")
        confirm = questionary.confirm("Do you still want to switch to this model?", default=False).ask()
        if not confirm:
            console.print("[dim]Model switch cancelled.[/dim]\n")
            return

    set_active_model(selected)
    console.print(f"[bold green]✔ Active AI Model switched to:[/bold green] [bold cyan]{selected}[/bold cyan]\n")


def handle_global_search():
    """
    Searches past chat interactions across all sessions.
    Renders results in a fixed 8-item scrolling window (scrollable with ↑/↓)
    and allows 1-click jump into the target session to continue chatting.
    """
    query = questionary.text("Enter keyword or question snippet to search across all sessions:").ask()
    if not query or not query.strip():
        return

    with console.status("[cyan]Searching conversation transcripts across all sessions...[/cyan]"):
        results = search_chat_history(query.strip(), limit=50)

    if not results:
        console.print(f"\n[yellow]No past interactions found matching '{query}'.[/yellow]\n")
        return

    console.print(f"\n[bold green]Found {len(results)} matching interaction(s) across all sessions:[/bold green]")

    # Build selectable list with 8-item fixed scrolling window
    choices = []
    for r in results:
        s_name = r.get("session_name", "Unknown")
        t_stamp = r.get("timestamp", "")
        short_q = r["user_query"].replace("\n", " ").strip()
        if len(short_q) > 60:
            short_q = short_q[:57] + "..."
        display_title = f"[{s_name}] \"{short_q}\" — {t_stamp}"
        choices.append(questionary.Choice(title=display_title, value=r))

    choices.append(questionary.Choice(title="< Back to Main Menu", value=None))

    selected_turn = questionary.select(
        "Select an interaction to view full transcript & context (Scroll with ↑/↓):",
        choices=choices
    ).ask()

    if not selected_turn or not isinstance(selected_turn, dict):
        return

    # Render full interaction details in a panel
    target_session = selected_turn.get("session_name", "")
    t_stamp = selected_turn.get("timestamp", "")
    m_used = selected_turn.get("model_used") or "Unknown"
    srcs = selected_turn.get("sources_summary") or "None"

    panel_content = (
        f"[bold cyan]Session Name:[/bold cyan] {target_session}\n"
        f"[bold cyan]Timestamp:[/bold cyan] {t_stamp} | [bold cyan]Model:[/bold cyan] `{m_used}`\n"
        f"[bold cyan]Sources:[/bold cyan] {srcs}\n\n"
        f"[bold]User Question:[/bold]\n{selected_turn['user_query']}\n\n"
        f"[bold magenta]AI Response:[/bold magenta]\n{selected_turn['ai_response']}"
    )
    console.print(Panel(panel_content, title=f"Transcript Match: {target_session}", border_style="cyan"))

    # Offer 1-click jump to session
    jump_confirm = questionary.confirm(
        f"Jump into session '{target_session}' and continue chatting with this document?",
        default=True
    ).ask()

    if jump_confirm:
        handle_chat(target_session=target_session)