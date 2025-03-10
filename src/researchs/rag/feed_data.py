from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import psycopg2
import json
from langchain_community.vectorstores import PGVector
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
                "lesson_id": lesson_id,
                "course_id": course_id,
                "course_name": course_name,
                "lesson_description": lesson_description,
                "course_description": course_description,
            }
            
            # doc = Document(page_content=content, metadata=metadata)
            documents.append({"page_content": content, "metadata": metadata})
        
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings():
    ''' Function to create vector embeddings '''
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    return ollama_embeddings

def stuff_vectordatabase(chunks, embeddings, collection_name, connection_str):
    ''' Function to load the chunks into the vector database '''
    docsearch = PGVector.from_documents(documents=chunks, embedding=embeddings, connection_string=connection_str, collection_name=collection_name)
    return docsearch

# Example usage
if __name__ == "__main__":
    connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"

    documents: List[Document] = []
    # lesson_documents = get_lessons_with_metadata(connection_string)
    # for doc in lesson_documents:
    #     document = Document(page_content=doc["page_content"], metadata=doc["metadata"])
    #     documents.append(document)
    # print(len(documents))
    # chunks = chunk_data(documents[:2])
    embeddings = create_embeddings()
    # vectorstore = stuff_vectordatabase(chunks=chunks,embeddings=embeddings,collection_name="lesson_content", connection_str=connection_string)
    vectorstore = PGVector(embedding_function=embeddings, collection_name="lesson_content", connection_string=connection_string)
    # print("### Done!")
    metadata_field_info = [
        AttributeInfo(
            name="lesson_id",
            description="The unique identifier for the lesson",
            type="string",
        ),
        AttributeInfo(
            name="course_id",
            description="The unique identifier for the course",
            type="string",
        ),
        AttributeInfo(
            name="course_name",
            description="The name of the course",
            type="string",
        ),
        AttributeInfo(
            name="lesson_description",
            description="A brief description of the lesson content",
            type="string",
        ),
        AttributeInfo(
            name="course_description",
            description="A detailed description of the course, including learning objectives and topics covered",
            type="string",
        ),
    ]
    
    # Examples for few-shot learning
    examples = [
        (
            "Summarize each lessons from course Matrix Data Structure Guide with course id 4e26b4bd-d406-4641-9d68-3ba8e1c39c97",
            {
                "query": "all lessons of course",
                "filter": 'and(in("course_name", ["Matrix Data Structure Guide"]),in("course_id", ["4e26b4bd-d406-4641-9d68-3ba8e1c39c97"]))',
            },
        ),
    ]
    

    document_content_description = "Lesson content about datastructure and algorithms techniques"
    
    # Create constructor prompt
    constructor_prompt = get_query_constructor_prompt(
        document_content_description,
        metadata_field_info,
        allowed_comparators=PGVectorTranslator.allowed_comparators,
        examples=examples,
    )
    
    llm = ChatOllama(model="llama3.2", temperature=0, base_url="http://localhost:11434")
    # Create query constructor
    output_parser = StructuredQueryOutputParser.from_components()  
   
    query_constructor = constructor_prompt | llm | output_parser

    # Initialize the Self-Query Retriever
    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        structured_query_translator=PGVectorTranslator(),
        search_kwargs={'k': 5}
    )


    # retriever = SelfQueryRetriever.from_llm(
    #     llm, vectorstore, document_content_description, metadata_field_info, verbose=True
    # )
    # response = retriever.invoke("What are the name of all lessons in The Logic Building Problems courses?")
    # print(response)
    template = '''You are a summary expert in summarizing lessons from courses about programming. Reply in a professional and friendly tone with each lesson is a bullet point starts with its lessons name and lesson content.
    
    Use this context to answer the question:
    {context}
    Question: {question}'''
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)

    # Create a chatbot Question & Answer chain from the retriever
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # Example user query
    query = "Summarize each lessons from course Matrix Data Structure Guide with course id 4e26b4bd-d406-4641-9d68-3ba8e1c39c97"

    # Perform the retrieval and generate the response
    response = rag_chain_with_source.invoke({query})

    # Display the response
    print(response)
