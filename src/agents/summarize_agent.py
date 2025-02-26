from logging import config
# from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
import psycopg2
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_core.tools import BaseTool, tool
from core.llm import get_model
from core.settings import settings
from langchain.agents import initialize_agent, AgentType
from langgraph.managed import RemainingSteps
from agents.llama_guard import LlamaGuardOutput
from langgraph.graph import END, MessagesState, StateGraph
from typing import Any
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.messages import BaseMessage, AIMessage, convert_to_messages
from langchain.agents import load_tools, create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import re
from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI
 
DB_CONNECTION_STRING = "postgresql://postgres:123456@host.docker.internal:5433/intellab-db"
MAX_STRING_LENGTH = 1000000
# DB_CONNECTION_STRING = "postgresql://postgres:123456@localhost:5433/intellab-db"

class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """
    course_name: str
    course_id: str
    response: str
    is_contained: bool
    
    
def get_schema(_):
    db = SQLDatabase.from_uri(DB_CONNECTION_STRING)  # Adjust as needed
    schema = db.get_table_info()
    return schema

# template = """
# You are tasked with summarizing lessons from a course based on a given course name. Use the provided table schema, question, SQL query, and SQL response to generate a natural language response.
# Do not need to tell the process, just return the narutal summarization response with at least 150 words and do not abbreviate
# Prompt Template:

# Schema:
# {schema}

# Task: Based on the table schema, question, SQL query, and SQL response:

# Generate the SQL query below by replacing the placeholder {course_id} with the actual course name provided by the user.
# SQL Query:
# SELECT lesson_name, content FROM lessons WHERE course_id = '{course_id}';

# Once the query is executed, use the query result ({response}) to summarize the lessons of {course_name} and generate a natural language response to the question.

# Output format: just return the summary content, not SQL generation

# {response}
# Question: Summarize all lessons for the course with name {course_name}.
# """
template = """
You are tasked with summarizing lessons from a course based on a given course name. Use the provided question, SQL query, and SQL response to generate a natural language response.
Do not need to tell the process, just return the narutal summarization response with AT LEAST 150 words
Prompt Template:

Task: Based on the table schema, question, SQL query, and SQL response:

Use the query result ({response}) to summarize the lessons of {course_name} and generate a natural language response to the question.

Output format: just return the summary content, not SQL generation

{response}
Question: Summarize all lessons for the course with name {course_name}.
"""
prompt_response = ChatPromptTemplate.from_template(template)

def run_query(course_id):
    db = SQLDatabase.from_uri(DB_CONNECTION_STRING)  # Adjust as needed
    query = f"SELECT lesson_name, content FROM lessons WHERE course_id = '{course_id}' LIMIT 3"
    return db.run(query)

# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# --- Utilites ---
def extract_course_info(input_string):
    # Regular expression to match the course name, ID, and regenerate flag
    pattern = r"course name: (.*?), id: (.*?), regenerate: (true|false)"
    
    # Search the string for matches
    match = re.search(pattern, input_string, re.IGNORECASE)
    
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


# ---- extract message node ----
def extract_message(state: AgentState) -> Literal["check_contained_summary", "generate"]:
    print("-------- EXTRACT MESSAGE ---------")
    message_content = state["messages"][-1].content
    extract_values = extract_course_info(message_content)
    print(extract_values)
    return {
        "course_name": extract_values["course_name"],
        "course_id": extract_values["course_id"],
        "regenerate": extract_values["regenerate"]
    }

# check summary content tool:
# contain -> finalize response, otherwise generate

# ----  check contained summary CONDITIONAL node----
def check_contained_summary(state: AgentState) -> Literal["retrieve_existing", "generate"]:
    print("-------- CHECK CONTAINED SUMMARY ---------")
    course_name = state['course_name']
    regenerate = state['regenerate']
    query = f"""
        SELECT summary_content
        FROM course_summary
        WHERE course_name = '{course_name}';
    """
    db = SQLDatabase.from_uri(DB_CONNECTION_STRING)  # Adjust as needed
    result = db.run(query)
    if result == '' or regenerate:
        print("-------- REGENERATE --------")
        return "generate"
    print("-------- RETRIEVE EXISTING --------")
    return "retrieve_existing"

# ---- retrieve existing summary content node ----
def retrieve_existing(state: AgentState):
    print("------- EXISTED --------")
    course_name = state['course_name']
    query = f"""
        SELECT summary_content
        FROM course_summary
        WHERE course_name = '{course_name}';
    """
    db = SQLDatabase.from_uri(DB_CONNECTION_STRING, max_string_length=MAX_STRING_LENGTH)  # Adjust as needed
    result = db.run(query)
    cleaned_string = re.sub(r"^\[\('", "", result)
    cleaned_string = re.sub(r"\',\)\]$", "", cleaned_string)
    return {"response": cleaned_string}

# ---- generate node ----
def generate(state: AgentState, config: RunnableConfig):
    print("-------- GENERATE ---------")
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    full_chain = (
        RunnablePassthrough.assign(
            schema=get_schema,
            response=lambda vars: run_query(vars["course_id"]),
        )
        | prompt_response
        | llm
        | StrOutputParser()
    )
    response = full_chain.invoke({"course_name": state["course_name"], "course_id": state["course_id"]})
    with open("hehet.txt", "w") as f:
        f.write(response)
    return {"response": response}
 
# compare new with existing content
# more informative and valuable -> store to db, otherwise ignore new content and get the existing
class ComparisonContent(BaseModel):
    """Binary score to assess the informative between generated content and existing content."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
    
CHECKING_SYSTEM = """
You are the grader system assessing whether the new summary {new_summary} of {course_name} is more informative and valuable than the existing summary content {existing_content}.
Give a binary score 'yes' or 'no', where 'yes' means that the answer is new summary content more informative and valuable than existing content.
"""

CHECKING_PROMPT = ChatPromptTemplate.from_template(CHECKING_SYSTEM)

# ---- retrieve existing summary CONDITIONAL node
def retrieve_existing_summary(state: AgentState, config: RunnableConfig) -> Literal["finalize_response", "store_summary"]:
    print("-------- Retrieve existing Summary ---------")
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    query = f"""
        SELECT summary_content
        FROM course_summary
        WHERE course_name = '{state['course_name']}';
    """
    db = SQLDatabase.from_uri(DB_CONNECTION_STRING, max_string_length=MAX_STRING_LENGTH)  # Adjust as needed
    result = db.run(query)
    cleaned_string = re.sub(r"^\[\('", "", result)
    cleaned_string = re.sub(r"\',\)\]$", "", cleaned_string)
    
    model = CHECKING_PROMPT | llm.with_structured_output(ComparisonContent)
    comparison_grade: ComparisonContent = model.invoke({"existing_content": cleaned_string, "new_summary": state["response"], "course_name": state["course_name"]})
    if comparison_grade.binary_score == "no":
        print("-------- NO - RESPONSE --------")
        return "finalize_response"
    else:
        print("-------- YES - STORE --------")
        return "store_summary"

# store content to db
# ---- store summary node ----
def store_summary(state: AgentState):
    print("-------- STORE SUMMARY --------")
    new_content = state["response"]
    course_name = state["course_name"]
    course_id = state["course_id"]
    
    query = f"""
        SELECT summary_content
        FROM course_summary
        WHERE course_name = '{course_name}';
    """
    db = SQLDatabase.from_uri(DB_CONNECTION_STRING)  # Adjust as needed
    result = db.run(query)
    if result == '':
        # No summary record exists for the course, so insert new content
        insert_query = f"""
            INSERT INTO course_summary (course_id, course_name, summary_content)
            VALUES ('{course_id}', '{course_name}', '{new_content}');
        """
        db.run(insert_query)
        print(f"New summary added for course: {course_name}.")
    else:
        # Summary record exists but content is NULL, so update it
        update_query = f"""
            UPDATE course_summary
            SET summary_content = '{new_content}'
            WHERE course_id = '{course_id}';
        """
        db.run(update_query)
        print(f"Summary updated for course: {course_name}.")
    # else:
    #     print(f"Summary already exists for course: {course_name}. No action needed.")
    return {"response": new_content}

# ---- finalize response node ----
def finalize_response(state: AgentState):
    print("---FINALIZING THE RESPONSE---")
    print(state["response"])
    return {"messages": [AIMessage(content=state["response"])]}


agent = StateGraph(AgentState)

agent.add_node("extract_message", extract_message)
agent.add_node("generate", generate)
agent.add_node("retrieve_existing", retrieve_existing)
agent.add_node("store_summary", store_summary)
agent.add_node("finalize_response", finalize_response)

agent.set_entry_point("extract_message")
agent.add_edge("retrieve_existing", "finalize_response")
agent.add_edge("store_summary", "finalize_response")
agent.add_edge("finalize_response", END)

agent.add_conditional_edges(
    "extract_message",
    check_contained_summary
)

agent.add_conditional_edges(
    "generate",
    retrieve_existing_summary
)


summarize_assistant = agent.compile()


# inputs = {"messages": [("human", "The Logic Building Problems")]}
# for output in summarize_assistant.stream(inputs):
#     print(output)
#     print("\n---\n")