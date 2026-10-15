from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from core.config import LLM_MODEL

def build_chat_chain(retriever):
    llm = ChatOllama(model=LLM_MODEL, temperature=0.0)

    system_prompt = (
        "You are a strict factual assistant. Answer the user's question ONLY using the provided context below. "
        "If the answer cannot be found in the context, say: 'I cannot find that information in the document.' "
        "Do NOT assume or extrapolate.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)