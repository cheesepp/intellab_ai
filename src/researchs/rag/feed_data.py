import os
from typing import List
import uuid
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import psycopg2
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_nomic import NomicEmbeddings

load_dotenv()  # Load biến môi trường từ .env


import re
def get_lessons_with_metadata(connection_string: str):
    """
    Connects to PostgreSQL, retrieves lesson and course data, 
    and returns a list of LangChain Documents with metadata.
    """
    query = """
    SELECT 
        l.lesson_id, l.content, l.description AS lesson_description, l.lesson_name,
        c.course_id, c.course_name, c.description AS course_description
    FROM lessons l
    JOIN courses c ON l.course_id = c.course_id
    where l.content is not null;
    """
    
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Process results into LangChain Documents
        documents = []

        print(f"Number of lessons retrieved: {len(rows)}")

        for row in rows:
            lesson_id, content, lesson_description, lesson_name, course_id, course_name, course_description = row
            
            metadata = {
                "lesson_name": lesson_name,
                # "course_id": course_id,
                "course_name": course_name,
                # "lesson_description": lesson_description,
            }
            
            # doc = Document(page_content=content, metadata=metadata)
            # Define the regex pattern
            pattern = re.compile(r'^(?!C\+\+$)([A-Za-z0-9\s#+*:]+)\s*(?:-+)?\s*\n*\`{3,}\n(.*?)\n\`{3,}$|https?:\/\/[^\s]+|^-+\n?', re.DOTALL | re.MULTILINE)
            # Remove the pattern from content
            cleaned_content = re.sub(pattern, '', content)
            cleaned_content = f"Course Name: {course_name}\nLesson Name: {lesson_name}\n\nContent:\n" + cleaned_content
            
            
            # # Create directory for the course if it doesn't exist
            # course_directory = os.path.join("/Users/mac/HCMUS/datn/agent-service-toolkit/src/researchs/lesson_pdfs", course_name)
            # if not os.path.exists(course_directory):
            #     os.makedirs(course_directory)

            # # Create Markdown file for each lesson within the course directory
            # markdown_file_path = os.path.join(course_directory, f"{lesson_name}.md")
            # with open(markdown_file_path, 'w') as md_file:
            #     md_file.write(cleaned_content)
            
            
            documents.append({"page_content": cleaned_content, "metadata": metadata})
        
        print(f"Number of documents created: {len(documents)}")
        
        return documents
        
    except Exception as e:
        print(f"Error when get lessons with metadata: {e}")
        return []
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            
from langchain.document_loaders import CSVLoader
def read_csv_data(database, connection_str, embeddings, collection_name="problems_and_courses"):
    if database not in ["courses", "problems"]:
        raise ValueError("Invalid database name! must be courses or problems!")
    current_directory = os.getcwd()
    print("current directory: ", current_directory)
    loader = CSVLoader(f"./src/documents/{database}.csv", encoding="windows-1252")
    data = loader.load()
    print(f"Loaded {len(data)} records from {database}.csv")
    
    # [print('data:', _) for _ in data]
    
    ids = [str(uuid.uuid4()) for _ in data]
    docsearch = PGVector.from_documents(documents=data, embedding=embeddings, connection=connection_str, collection_name=collection_name, ids=ids)
    return docsearch
    
def chunk_data(data):
    ''' Function to split documents into chunks '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000000)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings():
    ''' Function to create vector embeddings '''
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    return ollama_embeddings

def create_nomic_embeddings():
    ''' Function to create vector embeddings '''
    NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
    nomic_embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=NOMIC_API_KEY)
    return nomic_embeddings

def create_huggingface_embeddings():

    ''' Function to create vector embeddings '''
    model_name = "nomic-ai/nomic-embed-text-v1.5"
    # Thêm cấu hình bắt buộc
    model_kwargs = {
        "trust_remote_code": True,
        "device": "mps"  # Sử dụng Metal Performance Shaders trên Mac M1
    }
    
    encode_kwargs = {
        "normalize_embeddings": True,  # Bắt buộc theo docs của Nomic
        "batch_size": 32,  # Giảm batch size để tránh lỗi memory
        "truncate": True  # Tự động cắt văn bản dài
    }
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs  # Thêm dòng này
    )
    return hf_embeddings

def stuff_vectordatabase(chunks, embeddings, collection_name, connection_str):
    ''' Function to load the chunks into the vector database '''
    ids = [str(uuid.uuid4()) for _ in chunks]
    docsearch = PGVector.from_documents(documents=chunks, embedding=embeddings, connection=connection_str, collection_name=collection_name, ids=ids)
    return docsearch

# Example usage
if __name__ == "__main__":
    connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"

    documents: List[Document] = []
    lesson_documents = get_lessons_with_metadata(connection_string)
    for doc in lesson_documents:
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    print('documents length in main: ',len(documents))
    chunks = chunk_data(documents)
    # embeddings = create_embeddings()
    # embeddings = create_huggingfacecom_embeddings()
    embeddings = create_nomic_embeddings()
    print('STORING.......')
    vectorstore = stuff_vectordatabase(chunks=documents,embeddings=embeddings,collection_name="lesson_content", connection_str=connection_string)
    courses_data = read_csv_data("courses", connection_string, embeddings, "courses")
    problems_data = read_csv_data("problems", connection_string, embeddings, "problems")
    print('DONE!')