# ----------------- Dependencies -----------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import io
from typing import Dict
from chat_history import save_message, get_n_messages, clear_chat
print("ok1")
# ----------------- Embedding -----------------------
# embedding model -> multilingual-e5-small
# vector db -> Chroma db

embedder = TextEmbedding(
    model_name="intfloat/multilingual-e5-small"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200, 
)

db_location = "data/uploaded_files"
 
vector_store = Chroma(
    collection_name="demo",
    persist_directory=db_location,
    embedding_function=embedder)


retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

def extract_text_from_pdf(file_bytes:bytes) -> str:
    """
    Extracts text from a PDF given in form of bytes. To be Replaced by OCR system later.
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
print("ok2")

# ----------------- LLM -----------------------
# LLM -> Qwen2.5-1.5B-Instruct
# Server -> Llama.cpp server

USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

if USE_LOCAL_LLM:
    # llama.cpp config
    LLM_URL = os.getenv("LLM_URL")
    model = ChatOpenAI(
        model="...",
        temperature=0.2,
        # max_tokens=None,
        # timeout=None,
        # max_retries=None,
        api_key="...",
        base_url=f"{LLM_URL}/v1",
        streaming=True
    )
else:
    # Groq config
    model = ChatOpenAI(
        model="llama3-8b-8192",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        streaming=True,
    )

template = """
You are a RAG assistant. Your answers must be derived from the supplied context. Do not hallucinate or invent false facts. If the relavent information is missing or unclear in the context, respond only with: \"I don't know\". If neither history nor the relevant context is provided, respond only with \"No relevant context provided\".Provide responses that are accurate, explanatory and directly grounded in the context. Use chat history provided below for maintaining context.

Chat History : {history}
relevant text chunks : {chunks} 
user query : {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

def result(query:str, new_chat:bool):
    if new_chat == True:
        clear_chat()

    docs = retriever.invoke(query)
    chunks = "\n\n".join(d.page_content for d in docs)
    history = get_n_messages()

    full_response = ""

    for chunk in chain.stream({
        "history": history,
        "chunks": chunks,
        "question": query
    }):
        if chunk.content:
            full_response += chunk.content
            yield chunk.content
            
    save_message(query, full_response)
print("ok3")