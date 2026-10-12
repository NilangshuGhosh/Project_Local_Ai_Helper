from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the document
print("Loading document...")
loader = TextLoader("sample.txt",encoding='utf-8')
docs = loader.load()

# 2. Split it into chunks
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. Create embeddings and store in Chroma
print("Embedding and saving to database...")
# We use 'nomic-embed-text', a highly efficient embedding model available via Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    persist_directory="./chroma_db" # This creates a local folder to save the DB
)

print("Brain built successfully! Check the ./chroma_db folder.")