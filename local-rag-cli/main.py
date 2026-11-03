import sys
import warnings

# Clean terminal UI: Silence noisy internal framework deprecation notices
warnings.filterwarnings("ignore")
_orig_showwarning = warnings.showwarning
def _clean_showwarning(message, category, filename, lineno, file=None, line=None):
    msg_str = str(message)
    if "Chroma" in msg_str or "langchain" in msg_str.lower() or "deprecated" in msg_str.lower():
        return
    _orig_showwarning(message, category, filename, lineno, file=file, line=line)
warnings.showwarning = _clean_showwarning

import questionary
from rich.console import Console

from core.health import display_health_banner
from core.config import get_active_model
from tui.menu import (
    handle_list_sessions,
    handle_ingest,
    handle_chat,
    handle_manage_session_docs,
    handle_delete,
    handle_storage_health,
    handle_switch_model,
)

console = Console()


def main():
    console.print("\n[bold cyan]╔═══════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║       Local RAG Knowledge Assistant          ║[/bold cyan]")
    console.print("[bold cyan]╚═══════════════════════════════════════════════╝[/bold cyan]\n")

    # Run startup health check
    display_health_banner(console)

    while True:
        active_model = get_active_model()
        try:
            action = questionary.select(
                "What would you like to do?",
                choices=[
                    "Chat with a Document",
                    "Ingest Document (New or Existing Session)",
                    "Manage Documents in a Session (Inspect / Remove / Add)",
                    "List Active Sessions",
                    f"Switch Active AI Model (Current: {active_model})",
                    "Delete (Session or Document)",
                    "Storage & System Health",
                    "Exit"
                ]
            ).ask()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user. Exiting assistant. Goodbye![/yellow]")
            break

        if action == "Chat with a Document":
            handle_chat()
        elif action == "Ingest Document (New or Existing Session)":
            handle_ingest()
        elif action == "Manage Documents in a Session (Inspect / Remove / Add)":
            handle_manage_session_docs()
        elif action == "List Active Sessions":
            handle_list_sessions()
        elif action and action.startswith("Switch Active AI Model"):
            handle_switch_model()
        elif action == "Delete (Session or Document)":
            handle_delete()
        elif action == "Storage & System Health":
            handle_storage_health()
        elif action == "Exit" or action is None:
            console.print("[yellow]Exiting assistant. Goodbye![/yellow]")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
        sys.exit(0)