from typing import List
import uuid
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
            
from langchain.document_loaders import CSVLoader
import asyncio
connection_string = os.getenv("DB_CONNECTION_STRING")
#connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"

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
    cursor = None
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

        count: int = 0
        for row in rows:
            lesson_id, content, lesson_description, lesson_name, course_id, course_name, course_description = row
            
            # count += 1
            # if count > 5:
            #     break

            metadata = {
                "lesson_name": lesson_name,
                "course_id": course_id,
                "course_name": course_name,
                "lesson_id": lesson_id
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

def get_courses_with_metadata(connection_string: str):
    """
    Connects to PostgreSQL, retrieves course data, 
    and returns a list of LangChain Documents with metadata.
    """
    query = """
    SELECT 
        c.course_id, c.course_name, c.description AS course_description, 
        c.level, c.price,c.average_rating, c.review_count
    FROM courses c
    where c.description is not null;
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

        print(f"Number of courses retrieved: {len(rows)}")

        for row in rows:
            course_id, course_name, course_description, level, price, average_rating, review_count = row
            
            metadata = {
                "course_id": course_id,
                "course_name": course_name,
            }

            # Define the regex pattern
            pattern = re.compile(r'^(?!C\+\+$)([A-Za-z0-9\s#+*:]+)\s*(?:-+)?\s*\n*\`{3,}\n(.*?)\n\`{3,}$|https?:\/\/[^\s]+|^-+\n?', re.DOTALL | re.MULTILINE)
            # Remove the pattern from content
            cleaned_content = re.sub(pattern, '', course_description)
            cleaned_content = f"Course Id: {course_id}\nCourse Name: {course_name}\nCourse level: {level}\nCourse price: {price}\nAverage rating: {average_rating}\nReview count: {review_count}\n\nContent:\n" + cleaned_content

            documents.append({"page_content": cleaned_content, "metadata": metadata})

        print(f"Number of documents created: {len(documents)}")
        
        return documents
        
    except Exception as e:
        print(f"Error when get courses with metadata: {e}")
        return []
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_course_with_metadata_by_course_id(course_id: str, connection_string: str) -> List[Document]:
    """
    Connects to PostgreSQL, retrieves course data by course_id, 
    and returns a list of LangChain Documents with metadata.
    """
    query = """
    SELECT 
        c.course_id, c.course_name, c.description AS course_description, 
        c.level, c.price, c.average_rating, c.review_count
    FROM courses c
    WHERE c.course_id = %s;
    """

    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()

        # Execute query
        cursor.execute(query, (course_id,))
        rows = cursor.fetchall()

        # Process results into LangChain Documents
        documents = []

        for row in rows:
            course_id, course_name, course_description, level, price, average_rating, review_count = row

            metadata = {
                "course_id": course_id,
                "course_name": course_name
            }

            # Define the regex pattern
            pattern = re.compile(r'^(?!C\+\+$)([A-Za-z0-9\s#+*:]+)\s*(?:-+)?\s*\n*\`{3,}\n(.*?)\n\`{3,}$|https?:\/\/[^\s]+|^-+\n?', re.DOTALL | re.MULTILINE)
            # Remove the pattern from content
            cleaned_content = re.sub(pattern, '', course_description)
            cleaned_content = f"Course Id: {course_id}\nCourse Name: {course_name}\nCourse level: {level}\nCourse price: {price}\nAverage rating: {average_rating}\nReview count: {review_count}\n\nContent:\n" + cleaned_content

            documents.append({"page_content": cleaned_content, "metadata": metadata})

        return documents

    except Exception as e:
        print(f"Error when get course with course id: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_lessons_with_metadata_by_course_id(course_id: str, connection_string: str):
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
    where l.content is not null and c.course_id = %s;
    """
    
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query, (course_id,))
        rows = cursor.fetchall()
        
        # Process results into LangChain Documents
        documents = []

        print(f"Number of lessons retrieved: {len(rows)}")

        for row in rows:
            lesson_id, content, lesson_description, lesson_name, course_id, course_name, course_description = row
            
            metadata = {
                "lesson_name": lesson_name,
                "course_id": course_id,
                "course_name": course_name,
                "lesson_id": lesson_id
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
        print(f"Error when get lessons with metadata by course id: {e}")
        return []
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_problems_with_metadata(connection_string: str):
    """
    Connects to PostgreSQL, retrieves problem data, 
    and returns a list of LangChain Documents with metadata.
    """
    query = """
    SELECT 
        p.problem_id, p.problem_name, p.description, p.score, p.problem_level
    FROM problems p
    where p.description is not null;
    """
    cursor = None
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Process results into LangChain Documents
        documents = []

        print(f"Number of problems retrieved: {len(rows)}")

        count: int = 0
        for row in rows:
            problem_id, problem_name, description, score, problem_level = row
            
            # count += 1
            # if count > 5:
            #     break

            metadata = {
                "problem_id": problem_id,
                "problem_name": problem_name,
            }
            
            # doc = Document(page_content=content, metadata=metadata)
            # Define the regex pattern
            pattern = re.compile(r'^(?!C\+\+$)([A-Za-z0-9\s#+*:]+)\s*(?:-+)?\s*\n*\`{3,}\n(.*?)\n\`{3,}$|https?:\/\/[^\s]+|^-+\n?', re.DOTALL | re.MULTILINE)
            # Remove the pattern from content
            cleaned_content = re.sub(pattern, '', description)
            cleaned_content = f"Problem Name: {problem_name}\n\nContent:\n" + cleaned_content


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
        print(f"Error when get problems with metadata: {e}")
        return []
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def read_csv_data(database, connection_str, embeddings, collection_name="problems_and_courses"):
    if database not in ["courses", "problems"]:
        raise ValueError("Invalid database name! must be courses or problems!")
    current_directory = os.getcwd()
    print("current directory: ", current_directory)
    loader = CSVLoader(f"./src/documents/{database}.csv", encoding="windows-1252")
    data = loader.load()
    print(f"Loaded {len(data)} records from {database}.csv")
    ids = [str(uuid.uuid4()) for _ in data]
    docsearch = PGVector.from_documents(documents=data, embedding=embeddings, connection=connection_str, collection_name=collection_name, ids=ids)
    return docsearch
    
def update_csv_data(database, connection_str, embeddings, collection_name="problems_and_courses"):
    """
    Update (delete and re-insert) CSV data in the vector database.
    """
    if database not in ["courses", "problems"]:
        raise ValueError("Invalid database name! must be courses or problems!")
    current_directory = os.getcwd()
    print("current directory: ", current_directory)
    loader = CSVLoader(f"./src/documents/{database}.csv", encoding="windows-1252")
    data = loader.load()
    print(f"Loaded {len(data)} records from {database}.csv")
    ids = [str(uuid.uuid4()) for _ in data]
    
    # Connect to the vector store
    vectorstore = PGVector(
        connection=connection_str,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    # Delete old vectors for this CSV (if you have a metadata field to filter by)
    print(f"DELETING OLD VECTORS for {database}...")
    # If your CSVLoader adds a metadata field like 'source', use it. Otherwise, this will delete all.
    vectorstore.delete(filter={"source": database})
    
    # Add new/updated vectors
    print(f"STORING UPDATED VECTORS for {database}...")
    vectorstore.add_documents(data, ids=ids)
    print('DONE updating CSV data!')
    return

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
    ids = [str(uuid.uuid4()) for _ in chunks]
    docsearch = PGVector.from_documents(documents=chunks, embedding=embeddings, connection=connection_str, collection_name=collection_name, ids=ids)
    return docsearch

# Print all metadata in the vectorstore
def print_all_vectorstore_metadata(vectorstore):
    # Retrieve all documents (use a broad query, e.g., empty string)
    results = vectorstore.similarity_search("", k=1000)  # Adjust k as needed
    print(f"Total vectors found: {len(results)}")
    for doc in results:
        print(doc.metadata)

# Example usage
def feed_embedded_lesson_data() -> None:
    
    #connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"

    #connection_string = os.getenv("DB_CONNECTION_STRING")

    # print('connection string: ', connection_string)
    documents: List[Document] = []
    lesson_documents = get_lessons_with_metadata(connection_string)
    for doc in lesson_documents:
        #print('doc: ', doc["metadata"])
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    print('documents length in main: ',len(documents))
    chunks = chunk_data(documents)
    embeddings = create_embeddings()
    print('STORING.......')
    vectorstore = stuff_vectordatabase(chunks=documents,embeddings=embeddings,collection_name="lesson_content", connection_str=connection_string)
    #courses_data = read_csv_data("courses", connection_string, embeddings, "courses")
    #problems_data = read_csv_data("problems", connection_string, embeddings, "problems")
    print('DONE!')
    return

def feed_embedded_course_data() -> None:
    documents: List[Document] = []
    course_documents = get_courses_with_metadata(connection_string)
    for doc in course_documents:
        #print('doc: ', doc["metadata"])
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    print('documents length in main: ',len(documents))
    chunks = chunk_data(documents)
    embeddings = create_embeddings()
    print('STORING.......')
    vectorstore = stuff_vectordatabase(chunks=documents,embeddings=embeddings,collection_name="courses", connection_str=connection_string)
    #courses_data = read_csv_data("courses", connection_string, embeddings, "courses")
    print('DONE!')

def feed_embedded_problem_data() -> None:
    #connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"

    #connection_string = os.getenv("DB_CONNECTION_STRING")

    #print('connection string: ', connection_string)

    documents: List[Document] = []
    problem_documents = get_problems_with_metadata(connection_string)
    for doc in problem_documents:
        #print('doc: ', doc["metadata"])
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    print('documents length in main: ',len(documents))
    chunks = chunk_data(documents)
    embeddings = create_embeddings()
    print('STORING.......')
    vectorstore = stuff_vectordatabase(chunks=documents,embeddings=embeddings,collection_name="problems", connection_str=connection_string)
    print('DONE!')
    return

def get_lesson_with_metadata_by_lesson_id(lesson_id: str, connection_string: str) -> List[Document]:
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
    where l.content is not null and l.lesson_id = %s;
    """
    
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query, (lesson_id,))
        rows = cursor.fetchall()
        
        # Process results into LangChain Documents
        documents = []

        print(f"Number of lessons retrieved: {len(rows)}")

        for row in rows:
            lesson_id, content, lesson_description, lesson_name, course_id, course_name, course_description = row
            
            metadata = {
                "lesson_name": lesson_name,
                "course_id": course_id,
                "course_name": course_name,
                "lesson_id": lesson_id
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

def get_problem_with_metadata_by_problem_id(problem_id: str, connection_string: str) -> List[Document]:
    """
    Connects to PostgreSQL, retrieves lesson and course data, 
    and returns a list of LangChain Documents with metadata.
    """
    query = """
    SELECT 
        p.problem_id, p.problem_name, p.description, p.score, p.problem_level
    FROM problems p
    where p.description is not null and p.problem_id = %s;
    """
    
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query, (problem_id,))
        rows = cursor.fetchall()
        
        # Process results into LangChain Documents
        documents = []

        print(f"Number of lessons retrieved: {len(rows)}")

        for row in rows:
            problem_id, problem_name, description, score, problem_level = row
            
            metadata = {
                "problem_id": problem_id,
                "problem_name": problem_name,
            }
            
            # doc = Document(page_content=content, metadata=metadata)
            # Define the regex pattern
            pattern = re.compile(r'^(?!C\+\+$)([A-Za-z0-9\s#+*:]+)\s*(?:-+)?\s*\n*\`{3,}\n(.*?)\n\`{3,}$|https?:\/\/[^\s]+|^-+\n?', re.DOTALL | re.MULTILINE)
            # Remove the pattern from content
            cleaned_content = re.sub(pattern, '', description)
            cleaned_content = f"Problem Name: {problem_name}\n\nContent:\n" + cleaned_content


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

def embed_data_by_lesson_id(lesson_id: str) -> None:
    """
    Function to embed data by lesson_id
    """
    #connection_string = os.getenv("DB_CONNECTION_STRING")
    #connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"
    #connection_string = os.getenv("DB_CONNECTION_STRING")
    documents: List[Document] = []
    lesson_documents = get_lesson_with_metadata_by_lesson_id(lesson_id, connection_string)
    for doc in lesson_documents:
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    
    print('documents length in main: ',len(documents))
    if len(documents) <=0:
        print(f"No documents found for lesson_id: {lesson_id}. Skipping embedding.")
        return
    
    chunks = chunk_data(documents)
    embeddings = create_embeddings()
    print('STORING.......')
    vectorstore = stuff_vectordatabase(chunks=documents,embeddings=embeddings,collection_name="lesson_content", connection_str=connection_string)
    print('DONE!')

def embed_data_by_course_id(course_id: str) -> None:
    """
    Function to embed data by course_id
    """
    documents: List[Document] = []
    course_documents = get_course_with_metadata_by_course_id(course_id, connection_string)
    for doc in course_documents:
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    print('documents length in main: ',len(documents))

    if len(documents) <=0: 
        print(f"No documents found for course_id: {course_id}. Skipping embedding.")
        return

    chunks = chunk_data(documents)
    embeddings = create_embeddings()
    print('STORING.......')
    vectorstore = stuff_vectordatabase(chunks=documents,embeddings=embeddings,collection_name="courses", connection_str=connection_string)
    print('DONE!')

def embed_data_by_problem_id(problem_id: str) -> None:
    """
    Function to embed data by problem_id
    """
    #connection_string = os.getenv("DB_CONNECTION_STRING")
    #connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"
    #connection_string = os.getenv("DB_CONNECTION_STRING")
    documents: List[Document] = []
    lesson_documents = get_problem_with_metadata_by_problem_id(problem_id, connection_string)
    for doc in lesson_documents:
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    
    print('documents length in main: ',len(documents))
    if (len(documents) <=0):
        print(f"No documents found for problem_id: {problem_id}. Skipping embedding.")
        return
    
    chunks = chunk_data(documents)
    embeddings = create_embeddings()
    print('STORING.......')
    vectorstore = stuff_vectordatabase(chunks=documents,embeddings=embeddings,collection_name="problems", connection_str=connection_string)
    #courses_data = read_csv_data("courses", connection_string, embeddings, "courses")
    print('DONE!')

def delete_embeddings_by_lesson_id(
    lesson_id: str,
    connection_string: str =  os.getenv("DB_CONNECTION_STRING"), # "postgresql://postgres:123456@localhost:5433/intellab-db", #
    collection_name: str = "lesson_content",
    
) -> int:
    """
    Deletes embeddings from the langchain_pg_embedding table in a PostgreSQL database
    based on a specific lesson_id within the JSONB metadata of the document column.

    Args:
        connection_string: The PostgreSQL connection string.
        collection_name: The name of the Langchain collection to target.
        lesson_id: The specific lesson_id to match within the metadata.

    Returns:
        The number of deleted rows.

    Raises:
        psycopg2.Error: If there is an error executing the query.
    """
    try:
        conn = psycopg2.connect(connection_string)
            
        cursor = conn.cursor()

        query = f"""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid
                FROM langchain_pg_collection
                WHERE name = '{collection_name}'
            )
            AND cmetadata::jsonb ->> 'lesson_id' = '{lesson_id}';
        """

        cursor.execute(query)
        deleted_rows = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        return deleted_rows

    except psycopg2.Error as e:
        print(f"Error deleting embeddings: {e}")
        raise

def delete_lessons_embeddings_by_course_id(
    course_id: str,
    connection_string: str = os.getenv("DB_CONNECTION_STRING"), # "postgresql://postgres:123456@localhost:5433/intellab-db", #
    collection_name: str = "lesson_content",
    
) -> int:
    """
    Deletes embeddings from the langchain_pg_embedding table in a PostgreSQL database
    based on a specific course_id within the JSONB metadata of the document column.

    Args:
        connection_string: The PostgreSQL connection string.
        collection_name: The name of the Langchain collection to target.
        course_id: The specific course_id to match within the metadata.

    Returns:
        The number of deleted rows.

    Raises:
        psycopg2.Error: If there is an error executing the query.
    """
    try:
        conn = psycopg2.connect(connection_string)
            
        cursor = conn.cursor()

        query = f"""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid
                FROM langchain_pg_collection
                WHERE name = '{collection_name}'
            )
            AND cmetadata::jsonb ->> 'course_id' = '{course_id}';
        """

        cursor.execute(query)
        deleted_rows = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        return deleted_rows

    except psycopg2.Error as e:
        print(f"Error deleting lesson embeddings: {e}")
        raise

def delete_course_embeddings_by_course_id(
        course_id: str,
        connection_string: str = os.getenv("DB_CONNECTION_STRING"), # "postgresql://postgres:123456@localhost:5433/intellab-db", #os.getenv("DB_CONNECTION_STRING"),
        collection_name: str = "courses",
) -> int:
    """
    Deletes embeddings from the langchain_pg_embedding table in a PostgreSQL database
    based on a specific course_id within the JSONB metadata of the document column.

    Args:
        connection_string: The PostgreSQL connection string.
        collection_name: The name of the Langchain collection to target.
        course_id: The specific course_id to match within the metadata.

    Returns:
        The number of deleted rows.

    Raises:
        psycopg2.Error: If there is an error executing the query.
    """
    try:
        conn = psycopg2.connect(connection_string)

        cursor = conn.cursor()

        query = f"""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid
                FROM langchain_pg_collection
                WHERE name = '{collection_name}'
            )
            AND cmetadata::jsonb ->> 'course_id' = '{course_id}';
        """

        cursor.execute(query)
        deleted_rows = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        return deleted_rows

    except psycopg2.Error as e:
        print(f"Error deleting course embeddings: {e}")
        raise

def delete_embeddings_by_problem_id(
        problem_id: str,
        connection_string: str = os.getenv("DB_CONNECTION_STRING"), # "postgresql://postgres:123456@localhost:5433/intellab-db", #os.getenv("DB_CONNECTION_STRING"),
        collection_name: str = "problems",
) -> int:
    """
    Deletes embeddings from the langchain_pg_embedding table in a PostgreSQL database
    based on a specific problem_id within the JSONB metadata of the document column.

    Args:
        connection_string: The PostgreSQL connection string.
        collection_name: The name of the Langchain collection
        problem_id: The specific problem_id to match within the metadata.
    Returns:
        The number of deleted rows.

    Raises: 
        psycopg2.Error: If there is an error executing the query.
        Exception: If there is an unexpected error.
    """

    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()

        query = f"""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid
                FROM langchain_pg_collection
                WHERE name = '{collection_name}'
            )
            AND cmetadata::jsonb ->> 'problem_id' = '{problem_id}';
        """

        cursor.execute(query)
        deleted_rows = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        return deleted_rows

    except psycopg2.Error as e:
        print(f"Error deleting embeddings: {e}")
        raise

def delete_all_embeddings(
        connection_string: str = os.getenv("DB_CONNECTION_STRING"), # "postgresql://postgres:123456@localhost:5433/intellab-db", #os.getenv("DB_CONNECTION_STRING"),
) -> int:
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()

        query = f"DELETE FROM langchain_pg_embedding WHERE TRUE;"
        cursor.execute(query)
        deleted_rows = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        return deleted_rows

    except psycopg2.Error as e:
        print(f"Error deleting all embeddings: {e}")
        raise

def update_embedded_data_by_lesson_id(lesson_id: str) -> None:
    """
    Function to update embedded data by lesson_id
    """
    #connection_string = os.getenv("DB_CONNECTION_STRING")
    #connection_string = os.getenv("DB_CONNECTION_STRING")
    #connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"
    documents: List[Document] = []
    lesson_documents = get_lesson_with_metadata_by_lesson_id(lesson_id, connection_string)
    for doc in lesson_documents:
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    
    print('documents length in main: ',len(documents))
    if (len(documents) <=0):
        print(f"No documents found for lesson_id: {lesson_id}. Skipping update.")
        return

    chunks = chunk_data(documents)
    embeddings = create_embeddings()
    
    # 1. Connect to the vector store
    vectorstore = PGVector(
        connection=connection_string,
        collection_name="lesson_content",
        embeddings=embeddings,
        use_jsonb=True
    )

    #PGVector.delete(vectorstore, filter={"lesson_id": (lesson_id)})

    #print_all_vectorstore_metadata(vectorstore)

    # 2. Delete old vectors for this lesson (using lesson_id as filter)
    print('DELETING OLD VECTORS...')
    deleted_rows = delete_embeddings_by_lesson_id(
        lesson_id=lesson_id,
        connection_string=connection_string,
        collection_name="lesson_content"
    )
    print(f"Successfully deleted {(deleted_rows)} embeddings with lesson_id '{lesson_id}'.")

    # Retrieve documents with the specified lesson_id
    #print(f"lesson_id: {lesson_id}")
    # documents = vectorstore.similarity_search(
    #     query="*",  # Placeholder query
    #     k=1000,     # Adjust 'k' as needed
    #     filter={"lesson_id": lesson_id}
    # )

    # print(f"Number of vectors to delete: {len(documents)}")

    # for doc in documents:
    #     print(f'docs: {doc.metadata.get("uuid")}')

    # # Extract IDs from the retrieved documents
    # ids_to_delete = [doc.metadata["id"] for doc in documents if "id" in doc.metadata]


    # # Delete documents with the extracted IDs
    # vectorstore.delete(ids=ids_to_delete)

    # vectorstore.delete(filter={'lesson_id': (lesson_id)})
    # vectorstore.delete(filter={'lesson_id': lesson_id})

    # 3. Add new/updated vectors
    print('STORING UPDATED VECTORS...')
    vectorstore.add_documents(chunks)
    #vectorstore = stuff_vectordatabase(chunks=documents,embeddings=embeddings,collection_name="lesson_content", connection_str=connection_string)
    
    #update_csv_data("courses", connection_string, embeddings, "courses")
    print('DONE!')

def update_lessons_embedded_data_by_course_id(course_id: str) -> None:
    """
    Function to update embedded data by course_id
    """
    documents: List[Document] = []
    lesson_documents = get_lessons_with_metadata_by_course_id(course_id, connection_string)
    for doc in lesson_documents:
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    print('documents length in main: ',len(documents))

    if len(documents) <= 0:
        print(f"No documents found for course_id: {course_id}. Skipping update.")
        return
    
    chunks = chunk_data(documents)
    embeddings = create_embeddings()
    
    # 1. Connect to the vector store
    vectorstore = PGVector(
        connection=connection_string,
        collection_name="lesson_content",
        embeddings=embeddings,
        use_jsonb=True
    )

    #PGVector.delete(vectorstore, filter={"lesson_id": (lesson_id)})

    #print_all_vectorstore_metadata(vectorstore)

    # 2. Delete old vectors for this lesson (using lesson_id as filter)
    print('DELETING OLD VECTORS...')
    deleted_rows = delete_lessons_embeddings_by_course_id(
        course_id=course_id,
        connection_string=connection_string,
        collection_name="lesson_content"
    )
    print(f"Successfully deleted {(deleted_rows)} lessons embeddings with course_id '{course_id}'.")

    # Retrieve documents with the specified lesson_id
    #print(f"lesson_id: {lesson_id}")
    # documents = vectorstore.similarity_search(
    #     query="*",  # Placeholder query
    #     k=1000,     # Adjust 'k' as needed
    #     filter={"lesson_id": lesson_id}
    # )

    # print(f"Number of vectors to delete: {len(documents)}")

    # for doc in documents:
    #     print(f'docs: {doc.metadata.get("uuid")}')

    # # Extract IDs from the retrieved documents
    # ids_to_delete = [doc.metadata["id"] for doc in documents if "id" in doc.metadata]


    # # Delete documents with the extracted IDs
    # vectorstore.delete(ids=ids_to_delete)

    # vectorstore.delete(filter={'lesson_id': (lesson_id)})
    # vectorstore.delete(filter={'lesson_id':

    print('STORING UPDATED VECTORS...')
    vectorstore.add_documents(chunks)
    print('DONE!')

def update_course_embedded_data_by_course_id(course_id: str) -> None:
    """
    Function to update embedded data by course_id
    """
    documents: List[Document] = []
    course_documents = get_course_with_metadata_by_course_id(course_id, connection_string)
    for doc in course_documents:
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    
    print('documents length in main: ',len(documents))
    if len(documents) <= 0:
        print(f"No documents found for course_id: {course_id}. Skipping update.")
        return

    chunks = chunk_data(documents)
    embeddings = create_embeddings()

    # 1. Connect to the vector store
    vectorstore = PGVector(
        connection=connection_string,
        collection_name="courses",
        embeddings=embeddings,
        use_jsonb=True
    )

    print('DELETING OLD VECTORS...')
    deleted_rows = delete_course_embeddings_by_course_id(
        course_id=course_id,
        connection_string=connection_string,
        collection_name="courses"
    )
    print(f"Successfully deleted {(deleted_rows)} course embeddings with course_id '{course_id}'.")

    print('STORING UPDATED VECTORS...')
    vectorstore.add_documents(chunks)
    print('DONE!')

def update_embedded_data_by_problem_id(problem_id: str) -> None:
    """
    Function to update embedded data by problem_id
    """
    #connection_string = os.getenv("DB_CONNECTION_STRING")
    #connection_string = os.getenv("DB_CONNECTION_STRING")
    #connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"
    documents: List[Document] = []
    lesson_documents = get_problem_with_metadata_by_problem_id(connection_string, problem_id)
    for doc in lesson_documents:
        document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
        documents.append(document)
    
    print('documents length in main: ',len(documents))
    if (len(documents) <=0):
        print(f"No documents found for problem_id: {problem_id}. Skipping update.")
        return    

    chunks = chunk_data(documents)
    embeddings = create_embeddings()
    
    # 1. Connect to the vector store
    vectorstore = PGVector(
        connection=connection_string,
        collection_name="problems",
        embeddings=embeddings,
        use_jsonb=True
    )

    #PGVector.delete(vectorstore, filter={"lesson_id": (lesson_id)})

    #print_all_vectorstore_metadata(vectorstore)

    # 2. Delete old vectors for this lesson (using lesson_id as filter)
    print('DELETING OLD VECTORS...')
    deleted_rows = delete_embeddings_by_problem_id(
        problem_id=problem_id,
        connection_string=connection_string,
        collection_name="problems"
    )
    print(f"Successfully deleted {(deleted_rows)} embeddings with problem_id '{problem_id}'.")

    # Retrieve documents with the specified lesson_id
    #print(f"lesson_id: {lesson_id}")
    # documents = vectorstore.similarity_search(
    #     query="*",  # Placeholder query
    #     k=1000,     # Adjust 'k' as needed
    #     filter={"lesson_id": lesson_id}
    # )

    # print(f"Number of vectors to delete: {len(documents)}")

    # for doc in documents:
    #     print(f'docs: {doc.metadata.get("uuid")}')

    # # Extract IDs from the retrieved documents
    # ids_to_delete = [doc.metadata["id"] for doc in documents if "id" in doc.metadata]


    # # Delete documents with the extracted IDs
    # vectorstore.delete(ids=ids_to_delete)

    # vectorstore.delete(filter={'lesson_id': (lesson_id)})
    # vectorstore.delete(filter={'lesson_id': (lesson_id)})

    print('STORING UPDATED VECTORS...')
    vectorstore.add_documents(chunks)
    print('DONE!')

# if __name__ == "__main__":
#     # Example usage
#     # feed_embedded_data()
#     lesson_id = ('2faf87e7-73ca-41ba-b628-8460a26f8c14')
#     # embed_data_by_lesson_id(lesson_id)
#     update_embedded_data_by_lesson_id(lesson_id)
#     #asyncio.run(update_embedded_data_by_lesson_id(lesson_id))
#     pass