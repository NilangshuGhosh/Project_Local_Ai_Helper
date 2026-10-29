import typer
import os
import shutil
import sqlite3
import chromadb
from rich.console import Console
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

app = typer.Typer()
console = Console()
DB_DIR = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")


def cleanup_orphaned_folders():
    """Removes physical UUID folders from chroma_db that are no longer in sqlite."""
    db_file = os.path.join(DB_DIR, "chroma.sqlite3")
    if not os.path.exists(db_file):
        return
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("SELECT id FROM segments")
        active_segments = {row[0] for row in cur.fetchall()}
        conn.close()

        for item in os.listdir(DB_DIR):
            item_path = os.path.join(DB_DIR, item)
            if os.path.isdir(item_path) and item not in active_segments:
                shutil.rmtree(item_path, ignore_errors=True)
    except Exception:
        pass


@app.command()
def ingest(file_path: str, session_name: str):
    """Learn a new document and save it to a specific session."""
    if not os.path.exists(file_path):
        console.print(f"[red]Error: File {file_path} not found.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Studying {file_path}...[/cyan]")

    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=125)
    splits = text_splitter.split_documents(docs)

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=session_name,
    )

    console.print(f"[green]Done! Knowledge saved under session '{session_name}'.[/green]")
    console.print(f"Start chatting with: [bold]python cli.py chat {session_name}[/bold]")


@app.command()
def chat(session_name: str):
    """Start an interactive chat session about a specific document."""
    # 1. Guardrail: Verify session actually exists and is not empty
    client = chromadb.PersistentClient(path=DB_DIR)
    existing_collections = [c.name for c in client.list_collections()]

    if session_name not in existing_collections:
        console.print(f"[bold red]Error: Session '{session_name}' does not exist or was deleted.[/bold red]")
        console.print(f"[yellow]Run 'python cli.py ingest <file> {session_name}' first.[/yellow]")
        raise typer.Exit(code=1)

    collection = client.get_collection(session_name)
    if collection.count() == 0:
        console.print(f"[bold red]Error: Session '{session_name}' contains 0 documents.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Waking up brain for session: '{session_name}' ({collection.count()} chunks loaded)...[/cyan]")

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=session_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOllama(model="phi3", temperature=0.0)

    system_prompt = (
        "You are a strict factual assistant. Answer the user's question ONLY using the provided context below. "
        "If the answer cannot be found in the context, say: 'I cannot find that information in the document.' "
        "Do NOT assume, extrapolate, or bring in outside knowledge.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

    console.print("[bold green]Chat initialized! Type 'exit' to quit.[/bold green]")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        # Check retrieved context before calling the LLM
        retrieved_docs = retriever.invoke(user_input)
        if not retrieved_docs:
            console.print("\n[bold magenta]AI:[/bold magenta] I cannot find that information in the document.")
            continue

        console.print("[dim]Thinking...[/dim]")
        response = rag_chain.invoke({"input": user_input})
        console.print(f"\n[bold magenta]AI:[/bold magenta] {response['answer']}")

    # Cleanup prompt when exiting
    delete_intent = typer.confirm(f"\nDo you want to delete the knowledge for session '{session_name}'?")
    if delete_intent:
        client.delete_collection(session_name)
        cleanup_orphaned_folders()
        console.print(f"[yellow]Session '{session_name}' and its vector files have been completely deleted.[/yellow]")
    else:
        console.print("[green]Session saved for later![/green]")


if __name__ == "__main__":
    app()