import sys
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Check if user provided a question in the terminal
if len(sys.argv) < 2:
    print("Usage: python ask_brain.py 'Your question here'")
    sys.exit(1)
    
user_query = sys.argv[1]

# 1. Load the database and the embedding model
print("Waking up the brain...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Create a retriever that fetches the top 3 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2. Set up the LLM
llm = ChatOllama(model="phi3")

# 3. Create the Prompt Template
# This instructs the AI to ONLY use the provided context to answer.
system_prompt = (
    "You are a helpful AI assistant. Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4. Build the RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Execute the query
print(f"\nThinking about: '{user_query}'...\n")
response = rag_chain.invoke({"input": user_query})

print("--- Answer ---")
print(response["answer"])
print("\n--- Source Documents Used ---")
for doc in response["context"]:
    print(f"- {doc.page_content[:100]}...") # Print a snippet of what it read