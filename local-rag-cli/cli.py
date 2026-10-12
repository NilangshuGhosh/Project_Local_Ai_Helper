import typer
import os
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

@app.command()
def ingest(file_path: str, session_name: str):
    """Learn a new document and save it to a specific session."""
    if not os.path.exists(file_path):
        console.print(f"[red]Error: File {file_path} not found.[/red]")
        raise typer.Exit()

    console.print(f"[cyan]Studying {file_path}...[/cyan]")
    
    # Dynamically pick the right loader
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
        
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=125)
    splits = text_splitter.split_documents(docs)

    # Save to a SPECIFIC collection for this session
    Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=session_name
    )
    
    console.print(f"[green]Done! I've studied the file. You can now ask questions using:[/green]")
    console.print(f"python cli.py chat {session_name}")

@app.command()
def chat(session_name: str):
    """Start an interactive chat session about a specific document."""
    console.print(f"[cyan]Waking up brain for session: {session_name}...[/cyan]")
    
    # Load the specific collection
    vectorstore = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embeddings,
        collection_name=session_name
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOllama(model="phi3",temperature=0.0) # turned off creativity.

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
    
    # Interactive Chat Loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        console.print("[dim]Thinking...[/dim]")
        response = rag_chain.invoke({"input": user_input})
        console.print(f"\n[bold magenta]AI:[/bold magenta] {response['answer']}")

    # Cleanup prompt when exiting
    delete_intent = typer.confirm(f"\nDo you want to delete the knowledge for session '{session_name}'?")
    if delete_intent:
        vectorstore.delete_collection()
        console.print("[yellow]Session data deleted.[/yellow]")
    else:
        console.print("[green]Session saved for later![/green]")

if __name__ == "__main__":
    app()