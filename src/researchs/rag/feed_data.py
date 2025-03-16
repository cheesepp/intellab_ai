from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import psycopg2
import json
from langchain_postgres import PGVector
from langchain_community.document_loaders import JSONLoader
from pprint import pprint
from langchain_core.documents import Document
from langchain_community.query_constructors.pgvector import PGVectorTranslator
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import get_query_constructor_prompt
from langchain.chains.query_constructor.base import StructuredQueryOutputParser
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import re
import os
from fpdf import FPDF
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
    JOIN courses c ON l.course_id = c.course_id;
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
        
        # Write documents to a file
        with open('/Users/mac/HCMUS/datn/agent-service-toolkit/src/researchs/lesson_documents.json', 'w') as f:
            json.dump(documents, f, indent=4)
        
        return documents
        
    except Exception as e:
        print(f"Error: {e}")
        return []
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def chunk_data(data):
    ''' Function to split documents into chunks '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000000)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings():
    ''' Function to create vector embeddings '''
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    return ollama_embeddings

def stuff_vectordatabase(chunks, embeddings, collection_name, connection_str):
    ''' Function to load the chunks into the vector database '''
    docsearch = PGVector.from_documents(documents=chunks, embedding=embeddings, connection=connection_str, collection_name=collection_name)
    return docsearch

# Example usage
if __name__ == "__main__":
    connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"

    documents: List[Document] = []
    lesson_documents = get_lessons_with_metadata(connection_string)
    for doc in lesson_documents:
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    print(len(documents))
    chunks = chunk_data(documents)
    embeddings = create_embeddings()
    print('STORING.......')
    vectorstore = stuff_vectordatabase(chunks=documents,embeddings=embeddings,collection_name="lesson_content", connection_str=connection_string)
    print('DONE!')