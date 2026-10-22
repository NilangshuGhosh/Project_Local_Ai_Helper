import sys
import warnings

warnings.simplefilter("ignore")

import questionary
from rich.console import Console

from core.health import display_health_banner
from tui.menu import (
    handle_list_sessions,
    handle_ingest,
    handle_chat,
    handle_manage_session_docs,
    handle_delete,
    handle_storage_health,
)

console = Console()


def main():
    console.print("\n[bold cyan]╔═══════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║       Local RAG Knowledge Assistant          ║[/bold cyan]")
    console.print("[bold cyan]╚═══════════════════════════════════════════════╝[/bold cyan]\n")

    # Run startup health check
    display_health_banner(console)

    while True:
        try:
            action = questionary.select(
                "What would you like to do?",
                choices=[
                    "Chat with a Document",
                    "Ingest Document (New or Existing Session)",
                    "Manage Documents in a Session (Inspect / Remove / Add)",
                    "List Active Sessions",
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