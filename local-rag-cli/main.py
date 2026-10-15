import questionary
from rich.console import Console
from tui.menu import handle_list_sessions, handle_ingest, handle_chat, handle_delete

console = Console()

def main():
    console.print("[bold cyan]=== Local RAG Knowledge Assistant ===[/bold cyan]\n")
    
    while True:
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Chat with a Document",
                "Ingest New Document",
                "List Active Sessions",
                "Delete a Session",
                "Exit"
            ]
        ).ask()

        if action == "Chat with a Document":
            handle_chat()
        elif action == "Ingest New Document":
            handle_ingest()
        elif action == "List Active Sessions":
            handle_list_sessions()
        elif action == "Delete a Session":
            handle_delete()
        elif action == "Exit" or action is None:
            console.print("[yellow]Exiting assistant. Goodbye![/yellow]")
            break

if __name__ == "__main__":
    main()