from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import VectorStoreRetrieverMemory
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
import os
import io
from typing import Dict

# -----------------Vector database-----------------------
# embedding model -> all-minilm:l6-v2
# vector db -> Chroma db
embedder = OllamaEmbeddings(model="all-minilm:l6-v2")

db_location = "database"
db_flag = not os.path.exists(db_location)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200, 
)
        
vector_store = Chroma(
    collection_name="demo",
    persist_directory=db_location,
    embedding_function=embedder)


retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

def extract_text_from_pdf(file_bytes:bytes) -> str:
    """
    Extracts text from a PDF given in form of bytes.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def embed(pdfs:Dict[str, bytes]) -> None:
    """
    Embeds multiple PDFs given as (filename, file_bytes).
    """
    text = []
    for pdf in pdfs.items():
        text.append(Document(
            page_content=extract_text_from_pdf(pdf[1]),
            metadata={"source": pdf[0]}))
    docs = text_splitter.split_documents(text)
    ids = [str(i) for i in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=ids)
    return

def clear_db() -> None:
    """
    Truncates the contents of vector database.
    """
    ids = vector_store._collection.get()["ids"]
    if len(ids) >= 1:
        vector_store.delete(ids=ids)
    return
# -----------------LLM call-----------------------
# LLM -> LLaMa 3
# Server -> Ollama
model = OllamaLLM(model="llama3")

template = """
You are a RAG assistant. Your answers must be derived from the supplied context. Do not hallucinate or invent false facts. If the relavent information is missing or unclear in the context, respond only with: \"I don't know\". If neither history nor the relevant context is provided, respond only with \"No relevant context provided\".Provide responses that are accurate, explanatory and directly grounded in the context. Use chat history provided below for maintaining context.

Chat History : {history}
relevant text chunks : {chunks} 
user query : {question}
"""
prompt = ChatPromptTemplate.from_template(template)

conversation_store = Chroma(
    collection_name="chat_memory",
    embedding_function=embedder,
    persist_directory="./conversation_history")

memory = VectorStoreRetrieverMemory(
    retriever=conversation_store.as_retriever(), 
    memory_key="history", 
    input_key="question")

chain = prompt | model

def result(query:str, new_chat:bool) -> str:
    if new_chat == True:
        conversation_store.delete(ids=conversation_store._collection.get()["ids"])

    chunks = retriever.invoke(query)
    result = chain.invoke({
        "history":memory.load_memory_variables({"question": query})["history"],
        "chunks":chunks, 
        "question":query
        })
    memory.save_context({"input": query}, {"output": result})
    return result