from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
import os

embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
db_location = "chroma_db"
db_flag = not os.path.exists(db_location)

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200, 
)

if db_flag:
    df = extract_text_from_pdf("demo.pdf")
    doc = Document(page_content=df)
    docs = text_splitter.split_documents([doc])
    ids = [str(i) for i in range(len(docs))]
        
vector_store = Chroma(
    collection_name="demo",
    persist_directory=db_location,
    embedding_function=embeddings )

if db_flag: 
    vector_store.add_documents(documents=docs, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})