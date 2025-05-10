import operator
import os
import re
from typing import Literal
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
import psycopg2
from sqlalchemy import create_engine
from typing_extensions import Annotated, List, TypedDict
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.constants import Send
from langchain.sql_database import SQLDatabase
from langchain_core.messages import BaseMessage, AIMessage, convert_to_messages
from sqlalchemy import create_engine, text
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)

from core import settings
from core.llm import get_model

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# DB_CONNECTION_STRING = "postgresql://postgres:123456@localhost:5433/intellab-db"
# OLLAMA_HOST="http://localhost:11434"

def create_embeddings():
    ''' Function to create vector embeddings '''
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
    return ollama_embeddings

embeddings = create_embeddings()

vectorstore = PGVector(embeddings=embeddings, collection_name="lesson_content", connection=DB_CONNECTION_STRING, use_jsonb=True)


map_template = """You are an expert at summarizing programming education content. Your task is to create clear, concise summaries of individual lessons.

Lesson content:
{content}

Based on the provided lesson content:
1. Extract the lesson name from the section labeled "Lesson name:".
2. Identify the 3 most important concepts or techniques taught in the lesson.
3. Create a summary using the following format (start the response with 'Lesson'):

- Lesson: [LESSON NAME]
• [First key concept with a brief explanation]
• [Second key concept with a brief explanation]
• [Third key concept with a brief explanation]

Keep each bullet point concise and focused on one main idea. Include complexity analysis when relevant.
Ensure each bullet point is on its own line, not in a row.
Make the lesson name bold using markdown formatting.

Return the summary for this lesson.
"""
map_prompt = ChatPromptTemplate.from_template(map_template)

reduce_template = """
You are an expert at summarizing programming education content. You have received multiple lesson summaries for the course "{course_name}."

Your task is to provide a brief overall overview that highlights common themes or recurring techniques across the lessons.

Return the final comprehensive summary.
"""
reduce_prompt = ChatPromptTemplate.from_template(reduce_template)
MAX_STRING_LENGTH = 1000000

# llm = ChatOllama(model="llama3.2", temperature=0, base_url="http://localhost:11434")

# Define state for application
class State(MessagesState):
    context: List[Document]
    answer: str
    course_name: str
    course_id: str
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    regenerate: bool

token_max = 100000
def length_function(documents: List[Document], llm: BaseChatModel) -> int:
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

# --- Utilites ---
def extract_course_info(input_string):
    # Regular expression to match the course name, ID, and regenerate flag
    pattern = r"course name: (.*?), id: (.*?), regenerate: (true|false)"
    
    # Search the string for matches
    match = re.search(pattern, input_string, re.IGNORECASE)
    print(match)
    if match:
        # Extracted groups
        course_name = match.group(1)
        course_id = match.group(2)
        regenerate = match.group(3).lower() == 'true'  # Convert to boolean
        return {
            "course_name": course_name,
            "course_id": course_id,
            "regenerate": regenerate
        }
    else:
        raise ValueError("Input string does not match the expected format.")

def extract_message(state: State):
    print("-------- EXTRACT MESSAGE ---------")
    message_content = state["messages"][-1].content
    extract_values = extract_course_info(message_content)
    print(extract_values)
    return {
        "course_name": extract_values["course_name"],
        "course_id": extract_values["course_id"],
        "regenerate": extract_values["regenerate"]
    }
    
def retrieve(state: State):
    course_name = state["course_name"]
    retrieved_docs = vectorstore.as_retriever(search_kwargs={'k': 50, 'filter': {'course_name': course_name}})
    docs = retrieved_docs.invoke(course_name)
    return {"context": docs}

# check summary content tool:
# contain -> finalize response, otherwise generate

# Define your database connection
DB_ENGINE = create_engine(DB_CONNECTION_STRING)

# ---- Check Contained Summary Conditional Node ----
def check_contained_summary(state: dict) -> Literal["retrieve_existing", "retrieve"]:
    print("-------- CHECK CONTAINED SUMMARY ---------")
    
    course_name = state['course_name']
    regenerate = state.get('regenerate', False)  # Use `.get()` to avoid KeyError
    
    query = text("""
        SELECT summary_content
        FROM course_summary
        WHERE course_name = :course_name
    """)
    
    with DB_ENGINE.connect() as conn:
        result = conn.execute(query, {"course_name": course_name}).fetchone()
    
    if not result or result[0] == '' or regenerate:
        print("-------- REGENERATE --------")
        return "retrieve"
    
    print("-------- RETRIEVE EXISTING --------")
    return "retrieve_existing"

# ---- Retrieve Existing Summary Content Node ----
import re

def retrieve_existing(state: dict):
    print("------- EXISTED --------")
    
    course_name = state['course_name']
    
    query = text("""
        SELECT summary_content
        FROM course_summary
        WHERE course_name = :course_name
    """)

    with DB_ENGINE.connect() as conn:
        result = conn.execute(query, {"course_name": course_name}).fetchone()
    
    if result is None:
        return {"answer": ""}
    
    cleaned_string = re.sub(r"^\[\('", "", result[0])
    cleaned_string = re.sub(r"\',\)\]$", "", cleaned_string)
    
    return {"answer": cleaned_string}

# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    content: str
    
# Here we generate a summary, given a document
async def generate_summary(state: SummaryState, config: RunnableConfig):
    prompt = map_prompt.invoke(state["content"])
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await llm.ainvoke(prompt)
    return {"summaries": [response.content]}


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: State):
    print("============ MAP SUMMARIES ================")
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_summary", {"content": content}) for content in state["context"]
    ]

def collect_summaries(state: State):
    print("============ COLLECT SUMMARIES ================")
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }


async def _reduce(input: dict, llm: BaseChatModel) -> str:
    prompt = reduce_prompt.invoke(input)
    response = await llm.ainvoke(prompt)
    return response.content
  
# Add node to colbasechatlapse summaries
async def collapse_summaries(state: State):
    print("============ COLLAPSE SUMMARIES ================")
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    print(doc_lists)
    results = []
    for doc_list in doc_lists:
        print("-------- DOC LIST ------------")
        print(doc_list)
        results.append(await acollapse_docs(doc_list, _reduce))

    return {"collapsed_summaries": results}


# This represents a conditional edge in the graph that determines
# if we should collapse the summaries or not
def should_collapse(
    state: State,
    config: RunnableConfig
) -> Literal["collapse_summaries", "generate_final_summary"]:
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    num_tokens = length_function(state["collapsed_summaries"], llm=llm)
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"

# store content to db
# ---- store summary node ----
def store_summary(state: State):
    # """Insert a note into the PostgreSQL database."""
    # print("-------- STORE SUMMARY --------")
    new_content = state["answer"]
    # course_name = state["course_name"]
    # course_id = state["course_id"]
    # try:
    #     # Create a database engine
    #     engine = create_engine(DB_CONNECTION_STRING)
        
    #     with engine.connect() as conn:
    #         # Check if the course summary exists
    #         query = text("SELECT summary_content FROM course_summary WHERE course_id = :course_id")
    #         result = conn.execute(query, {"course_id": course_id}).scalar()

    #         if result is None:
    #             # Insert new summary
    #             insert_query = text("""
    #                 INSERT INTO course_summary (course_id, course_name, summary_content)
    #                 VALUES (:course_id, :course_name, :new_content)
    #             """)
    #             conn.execute(insert_query, {"course_id": course_id, "course_name": course_name, "new_content": new_content})
    #             print(f"New summary added for course: {course_name}.")
    #         elif result == '':
    #             # Update existing record if summary_content is empty
    #             update_query = text("""
    #                 UPDATE course_summary
    #                 SET summary_content = :new_content
    #                 WHERE course_id = :course_id
    #             """)
    #             conn.execute(update_query, {"course_id": course_id, "new_content": new_content})
    #             print(f"Summary updated for course: {course_name}.")
    #         else:
    #             print(f"Summary already exists for course: {course_name}. No action needed.")
            
    #         conn.commit()
    
    # except Exception as e:
    #     print("Database error:", e)
    return {"answer": new_content}
    
# Here we will generate the final summary
async def generate_final_summary(state: State, config: RunnableConfig):
    print("-------- GENERATE FINAL SUMMARY --------")
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await _reduce(state["collapsed_summaries"], llm=llm)
    summaries = state["collapsed_summaries"]
    combine_summary = "\n\n".join([summary.page_content for summary in summaries])
    combine_summary = combine_summary + f"\n\n{response}"
    return {"answer": combine_summary}

# ---- finalize response node ----
def finalize_response(state: State):
    print("---FINALIZING THE RESPONSE---")
    print(state["answer"])
    return {"messages": [AIMessage(content=state["answer"])]}

# Construct the graph
# Nodes:
graph = StateGraph(State)
graph.add_node("extract_message", extract_message)
graph.add_node("retrieve", retrieve)
graph.add_node("generate_summary", generate_summary)  # same as before
graph.add_node("collect_summaries", collect_summaries)
# graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)
graph.add_node("retrieve_existing", retrieve_existing)
graph.add_node("store_summary", store_summary)
graph.add_node("finalize_response", finalize_response)

# Edges:
graph.add_edge(START, "extract_message")
# graph.add_edge("extract_message", "retrieve")
graph.add_conditional_edges(
    "extract_message",
    check_contained_summary
)
graph.add_conditional_edges("retrieve", map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_edge("collect_summaries", "generate_final_summary")
# graph.add_conditional_edges("collect_summaries", should_collapse)
# graph.add_conditional_edges("collapse_summaries", should_collapse)
# graph.add_edge("generate_final_summary", END)
graph.add_edge("generate_final_summary", "store_summary")
graph.add_edge("retrieve_existing", "finalize_response")
graph.add_edge("store_summary", "finalize_response")
graph.add_edge("finalize_response", END)

summarize_assistant = graph.compile()