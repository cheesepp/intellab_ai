import re
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
import os
from langchain_core.runnables import (
    RunnableLambda,
    RunnableConfig,
)
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Literal, Annotated, TypedDict
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import (
    AnyMessage,
)
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, SystemMessage

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment, parse_llama_guard_output, parse_llama_guard_relevant_topic
from core import settings
from core.llm import get_model
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableParallel
from langchain_core.documents import Document

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# DB_CONNECTION_STRING = "postgresql://postgres:123456@localhost:5433/intellab-db"
# OLLAMA_HOST="http://localhost:11434"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

def create_embeddings():
    ''' Function to create vector embeddings '''
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
    return ollama_embeddings

vectorstore = PGVector(embeddings=create_embeddings(), collection_name="lesson_content", connection=DB_CONNECTION_STRING, use_jsonb=True)

# We will add a `summary` attribute (in addition to `messages` key,
# which MessagesState already has)
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    safety: LlamaGuardOutput
    is_relevant: LlamaGuardOutput
    lesson: str
    lesson_name: str
    question: str

# template = """
# You are an Expert Educational Content Interpreter specializing in explaining lesson materials. Your primary role is to provide clear, precise explanations of specific sentences or concepts from lesson content that users ask about.

# CONTEXT
# Lesson Content: {lesson_content}
# Current discussion: {conversation}

# USER QUERY
# The user is asking about: "{question}"

# RESPONSE GUIDELINES:
# 1. Focus specifically on explaining the sentence or concept the user is asking about
# 2. Connect the explanation directly to the broader lesson content provided in the context
# 3. Provide precise definitions of technical terms appearing in the sentence
# 4. Explain how this specific concept fits within the larger framework of the lesson
# 5. Use clear, concise language appropriate for educational settings
# 6. Include relevant examples from the lesson content when helpful
# 7. Maintain professional academic tone throughout your response
# 8. Answer directly without meta-references to "the context" or "the materials"

# When responding, first identify the specific sentence/concept in question, then provide a thorough explanation that places it within the broader context of the lesson material."""
template = """
You are an Expert Educational Content Interpreter specializing in explaining lesson materials. Your primary role is to provide clear, precise explanations of specific sentences or concepts from lesson content that users ask about.

CONTEXT
Lesson Content: {lesson_content}
Current discussion: {conversation}

USER QUERY
The user is asking about: "{question}"

RESPONSE GUIDELINES:
1. Focus specifically on explaining the sentence or concept the user is asking about
2. Connect the explanation directly to the broader lesson content provided in the context
3. Provide precise definitions of technical terms appearing in the sentence
4. Explain how this specific concept fits within the larger framework of the lesson
5. Use clear, concise language appropriate for educational settings
6. Include relevant examples from the lesson content when helpful
7. Maintain professional academic tone throughout your response
8. Answer directly without meta-references to "the context" or "the materials"
9. Limit responses to maximum 100 words
10. Be concise and focused on the core question - avoid lengthy explanations
11. Don't start responses with phrases like "Here...", "Because user has asked...", "According to the lesson context,...", or "The specific concept being asked about is:..."
12. If question is outside the lesson scope, immediately state it's beyond scope and invite questions related to the lesson
13. Present all information in plain text format only - do not use tables or code blocks
When responding, identify the specific sentence/concept and provide a concise explanation that places it within the broader context of the lesson material."""

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_HOST
)
# Create an index using the loaded documents
prompt = ChatPromptTemplate.from_template(template)

def format_is_relevant_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"{safety.unsafe_response}"
    )
    return AIMessage(content=content)

async def llama_guard_output(state: State, config: RunnableConfig) -> State:
    # llama_guard = LlamaGuard()
    # safety_output = await llama_guard.ainvoke("User", state["messages"])
    # return {"output_safety": safety_output}
    checking_template = """You are an expert at determining whether an input is relevant to programming or casual conversation that should be allowed in a programming-focused chatbot.

    User input: {question}

    Your task is to:
    1. Analyze the input carefully
    2. Determine if it's relevant to programming topics OR is appropriate casual conversation
    3. Provide a clear response

    For each input, respond with:
    - FIRST LINE: Either "relevant" or "irrelevant" (lowercase, no additional words)
    - SECOND LINE: A brief explanation of your decision, respond friendly and advise user back to programming topics, start with "Sorry, but..." if the input is irrelevant

    Classify as RELEVANT:
    - Programming languages and their features
    - Algorithms and data structures
    - Software development practices
    - Computer science concepts
    - Coding problems and their solutions
    - System design and architecture
    - Development tools and environments
    - Questions about programming courses or learning resources
    - Greetings and casual conversation starters (like "Hi", "How are you?", "Good morning")
    - Follow-up questions that might appear irrelevant in isolation but are likely part of a programming conversation
    - Questions about career advice in programming/software development
    - Any ambiguous question that could reasonably relate to programming

    Examples:
    Input: "How do I implement a binary search in Python?"
    Output:
    relevant
    This question directly relates to implementing an algorithm in a programming language.

    Input: "Hi there! Can you help me with some coding questions?"
    Output:
    relevant
    This is a greeting and conversation starter appropriate for a programming-focused chatbot.

    Input: "What's the best recipe for chocolate chip cookies?"
    Output:
    irrelevant
    This question is about cooking/baking and has no connection to programming.

    Input: "Can you recommend some resources for learning React?"
    Output:
    relevant
    This question is about learning resources for a programming framework.

    Input: "What do you think about politics today?"
    Output:
    irrelevant
    This question is about politics and not related to programming."""
    
    prompt = ChatPromptTemplate.from_template(checking_template)
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    checking_output = prompt | model
    messages = state["messages"]
    response = await checking_output.ainvoke({"question": messages[-1].content})
    llama_guard = parse_llama_guard_relevant_topic(response.content)
    return {"is_relevant": llama_guard}

async def block_irrelevant_content(state: State, config: RunnableConfig) -> State:
    safety: LlamaGuardOutput = state["is_relevant"]
    return {"messages": [format_is_relevant_message(safety)]}

# Check for unsafe input and block further processing if found
def check_is_relevant(state: State) -> Literal["relevant", "irrelevant"]:
    is_relevant: LlamaGuardOutput = state["is_relevant"]
    match is_relevant.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "irrelevant"
        case _:
            return "relevant"
        
def extract_message(state: State):
    message = state["messages"][-1].content
    print (f"========= GO {message} ============")
    pattern = r"Lesson:\s*(.*?)\s*Lesson_id:\s*(\S+)\s*Question:\s*(.+)"

    match = re.search(pattern, message, re.DOTALL)
    print (f"========= MATCH {match} ============")
    if match:
        lesson_name = match.group(1)
        lesson_id = match.group(2)
        question = match.group(3)
        print("Lesson Name:", lesson_name)
        print("Lesson ID:", lesson_id)
        print("Question:", question)
        return {"lesson_name": lesson_name, "question": question}
    else:
        return {"lesson_name": "", "question": ""}

def retrieve(state: State):
    lesson_name = state["lesson_name"]
    retrieved_docs = vectorstore.as_retriever(search_kwargs={'k': 1, 'filter': {'lesson_name': lesson_name}})
    docs = retrieved_docs.invoke(lesson_name)
    print(f"======== LESSON NAME: {lesson_name} DOCS: {docs}==========")
    return {"lesson": docs[0].page_content if len(docs) > 0 else ""}


# Define the logic to call the model
async def acall_model(state: State, config: RunnableConfig):
    lesson = state["lesson"]
    question = state["question"]
    
    print(f"======== {lesson} ==========")
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    # model = ChatOllama(model="codeqwen", base_url=OLLAMA_HOST)
    print(f"------- CALL MODEL ----------")
    # If a summary exists, we add this in as a system message
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    m_as_string = "\n\n".join([message.content for message in messages])
    normal_chatbot = {
        "conversation": RunnableLambda(lambda _: m_as_string),
        "lesson_content": RunnableLambda(lambda _: lesson),
        "question": RunnableLambda(lambda _: question),
      } | prompt | model
    response = await normal_chatbot.ainvoke(messages)
    return {"messages": [response]}

# We now define the logic for determining whether to end or summarize the conversation
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    # Otherwise we can just end
    return END


async def summarize_conversation(state: State, config: RunnableConfig):
    print(f"------- SUMMARIZE CONVERSATION ----------")
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    # First, we summarize the conversation
    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    summary_messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(summary_messages)
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
    

# Define a new graph
# Define the graph
agent = StateGraph(State)
agent.add_node("extract", extract_message)
agent.add_node("retrieve", retrieve)
agent.add_node("model", acall_model)
agent.add_node("guard_output", llama_guard_output)
agent.add_node(summarize_conversation)
agent.set_entry_point("extract")
agent.add_node("block_irrelevant_content", block_irrelevant_content)

agent.add_conditional_edges(
    "guard_output", check_is_relevant, {"irrelevant": "block_irrelevant_content", "relevant": "retrieve" }
)

agent.add_edge("extract", "guard_output")
agent.add_edge("retrieve", "model")
agent.add_edge("model", "summarize_conversation")
agent.add_edge("summarize_conversation", END)


lesson_chatbot = agent.compile()