from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

OLLAMA_HOST="http://localhost:11434"
DB_CONNECTION_STRING = "postgresql://postgres:123456@localhost:5433/intellab-db"

def create_embeddings():
    ''' Function to create vector embeddings '''
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
    return ollama_embeddings

vectorstore = PGVector(embeddings=create_embeddings(), collection_name="lesson_content", connection=DB_CONNECTION_STRING, use_jsonb=True)

def retrieve():
    retrieved_docs = vectorstore.as_retriever(search_kwargs={'k': 1, 'filter': {'lesson_name': "Introduction to Circular Linked List"}})
    docs = retrieved_docs.invoke("Introduction to Circular Linked List")
    return {"lesson": docs}
print(retrieve())