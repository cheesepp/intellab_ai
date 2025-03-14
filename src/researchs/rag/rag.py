from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
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
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

def create_embeddings():
    ''' Function to create vector embeddings '''
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    return ollama_embeddings

connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"
embeddings = create_embeddings()

vectorstore = PGVector(embeddings=embeddings, collection_name="lesson_content", connection=connection_string)
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
        name="lesson_name",
        description="The name of the lesson",
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
        "all lessons of course names The Logic Building Problems",
        {
            "query": "get all lessons where course_name equals The Logic Building Problems",
            "filter": 'in("course_name", ["The Logic Building Problems"])',
        },
    ),
    (
        "all lessons of course names Stack",
        {
            "query": "get all lessons where course_name equals Stack",
            "filter": 'in("course_name", ["Stack"])',
        },
    ),
    (
        "all lessons of course names Queue",
        {
            "query": "get all lessons where course_name like Queue",
            "filter": 'in("course_name", ["Queue"])',
        },
    ),
    (
        "all lessons of course names Stack",
        {
            "query": "get all lessons where course_name contains Stack",
            "filter": 'in("course_name", ["Stack"])',
        },
    ),
]


document_content_description = "Lesson content about datastructure and algorithms techniques"
llm = ChatOllama(model="llama3.2", temperature=0, base_url="http://localhost:11434")
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# # Create constructor prompt
constructor_prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
    allowed_comparators=PGVectorTranslator.allowed_comparators,
    examples=examples,
)

# # Create query constructor
output_parser = StructuredQueryOutputParser.from_components(
    allowed_comparators=PGVectorTranslator.allowed_comparators,
    allowed_operators=PGVectorTranslator.allowed_operators,
)  

query_constructor = constructor_prompt | llm | output_parser

# Initialize the Self-Query Retriever
retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=PGVectorTranslator(),
    search_kwargs={'k': 100}
)
# retriever = SelfQueryRetriever.from_llm(
#     llm=llm, 
#     vectorstore=vectorstore, 
#     document_contents=document_content_description, 
#     metadata_field_info=metadata_field_info, 
#     structured_query_translator=PGVectorTranslator(),
#     verbose=True,
#     search_kwargs={'k': 5}
# )
response = retriever.invoke('all lessons of course names Matrix Data Structure Guide')
print(response)
exit()
# response = retriever.invoke("What are the name of all lessons in The Logic Building Problems courses?")
# print(response)
# template = """You are an expert at summarizing programming educational content. Your task is to create clear, concise summaries of multiple programming lessons from various courses.

# Based on user request: summary {request}

# Lesson Context:
# {context}

# Instructions:
# 1. Identify each separate lesson in the provided context (each lesson typically begins with "Course Name:" and "Lesson Name:" headings)
# 2. For EACH lesson found:
#    a. Extract the course name and lesson name
#    b. From the "Content:" section, identify the 3 most important concepts or techniques taught
#    c. Create a summary with the lesson name followed by at most 3 bullet points

# Format EACH lesson summary as:
# **Lesson: [LESSON NAME]**
# - [First key concept with brief explanation]
# - [Second key concept with brief explanation]
# - [Third key concept with brief explanation]

# Present all lesson summaries in sequence, with clear separation between different lessons.

# Guidelines:
# - Each bullet point should focus on one main concept, algorithm, or approach
# - Include time and space complexity analysis when mentioned
# - Keep explanations concise but informative
# - Highlight implementation techniques and optimization strategies when present
# - For mathematical concepts, include the formal definition or formula"""

template = """You are an expert at summarizing programming education content. Your task is to create clear, concise summaries of programming lessons that capture the essential concepts.
Based on user request: summary {request}
Lesson Context:
{context}

For each lesson in the provided content:
1. Extract the lesson name
2. Identify the 3 most important concepts or techniques taught in the lesson
3. Create a summary with the lesson name followed by at most 3 bullet points

Format your summary as:
- Lesson: [LESSON NAME]
  • [First key concept with brief explanation]
  • [Second key concept with brief explanation]
  • [Third key concept with brief explanation]

Keep each bullet point concise and focused on one main idea. Include complexity analysis when relevant."""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    print(f"============= DOCS {docs} ===============")
    # formatted_doc = "\n\n".join(f"Course name: {doc.metadata['course_name']}\nLesson name: {doc.metadata['lesson_name']}\n\nContent: {doc.page_content}" for doc in docs)
    formatted_doc = "\n\n".join(f"{doc.page_content}" for doc in docs)
    with open("formatted_doc.txt", "w") as f:
        f.write(formatted_doc)
    return formatted_doc


# print(f'================ RETRIEVER {retriever} ==================')
query = "all lessons of course names Matrix Data Structure Guide"

# Create a chatbot Question & Answer chain from the retriever
rag_chain_from_docs = (
    {"context": retriever, "request": RunnableLambda(lambda _: query) }
    |
    RunnablePassthrough.assign(
        context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

# rag_chain_with_source = RunnableParallel(
#     {"context": retriever, "question": RunnablePassthrough()}
# ).assign(answer=rag_chain_from_docs)

# Example user query

# Perform the retrieval and generate the response
response = rag_chain_from_docs.invoke(query)

# Display the response
print(response)
# Convert the response dictionary to a JSON-formatted string
# Ensure that the 'query' key is a string or list
# response["question"] = list(response["question"]) if isinstance(response["question"], set) else response["question"]

# response_text = json.dumps(response, indent=4)

# Write the text to a file
with open("hehe.txt", "w") as f:
    f.write(response)