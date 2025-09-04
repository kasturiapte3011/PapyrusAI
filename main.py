from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3")
template = """
You are an agent thet helps user chat with their documents. You are provided with relevent chunks of text from the documents and you answer user query based on the given text chunks.

relevent text chunks : {chunks} 
user query : {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    query = input("Ask your question (enter q to quit):")
    if query == 'q':
        break
    
    chunks = retriever.invoke(query)
    result = chain.invoke({"chunks":chunks, "question":query})
    print(result) 