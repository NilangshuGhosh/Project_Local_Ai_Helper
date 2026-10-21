import sys
import questionary
from rich.console import Console

from core.health import display_health_banner
from tui.menu import (
    handle_list_sessions,
    handle_ingest,
    handle_chat,
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
                    "Ingest New Document",
                    "List Active Sessions",
                    "Delete a Session (with physical cleanup)",
                    "Storage & System Health",
                    "Exit"
                ]
            ).ask()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user. Exiting assistant. Goodbye![/yellow]")
            break

        if action == "Chat with a Document":
            handle_chat()
        elif action == "Ingest New Document":
            handle_ingest()
        elif action == "List Active Sessions":
            handle_list_sessions()
        elif action == "Delete a Session (with physical cleanup)":
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