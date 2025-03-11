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
from typing import Literal, Annotated, TypedDict
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

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# We will add a `summary` attribute (in addition to `messages` key,
# which MessagesState already has)
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    safety: LlamaGuardOutput
    is_relevant: LlamaGuardOutput
# We will use this model for both the conversation and the summarization
# model = ChatOllama(model="llama3.2")

# template = """You are an assistant for question-answering tasks that focus on algorithms and data structures. Use the following pieces of retrieved context to answer the question. Each question, you must explain deeply and understandable. If you don't know the answer, just say that you don't know. 
# Answer questions based on conversation history:
# Summary: {summary}
# Current conversation: {conversation}

# When you recommend some courses, always give the url along which point to that course with endpoint is course_id and appended with "https://localhost:3000/courses/" (just when recommending).
# Please follow strictly with the provided context, do not recommend any courses outside.
# Question: {question} 
# Context: {context} 
# Answer: Just response the question, do not say 'Based on' or something similar."""
template = """
You are an Educational Algorithm Assistant specializing in algorithms and data structures. Your responses should be thorough, clear, and educational.

CONVERSATION HISTORY
Summary: {summary}
Current discussion: {conversation}

QUESTION
{question}

REFERENCE MATERIALS
{context}

RESPONSE GUIDELINES:
1. Provide comprehensive explanations that build fundamental understanding
2. Include examples to illustrate complex concepts
3. Break down algorithmic approaches step-by-step
4. Discuss time/space complexity when relevant
5. When recommending problems, use the format: [Problem Title](https://localhost:3000/problems/problem_id)
6. When recommending courses, use the format: [Course Title](https://localhost:3000/courses/course_id)
7. Only recommend courses mentioned in the provided context
8. If you cannot answer based on the provided materials, acknowledge this clearly

Answer directly without phrases like "Based on the context" or "According to the materials."""

loader = CSVLoader(file_path='./documents/courses.csv')
problem_loader = CSVLoader(file_path='./documents/problems.csv')
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)
# Create an index using the loaded documents
index_creator = VectorstoreIndexCreator(embedding=embeddings)
docsearch = index_creator.from_loaders([loader, problem_loader])
prompt = ChatPromptTemplate.from_template(template)

# Define the logic to call the model
async def acall_model(state: State, config: RunnableConfig):
    print(f'------- CALL MODEL {config["configurable"].get("model", settings.DEFAULT_MODEL)}----------')
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    # If a summary exists, we add this in as a system message
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    # print(f"--------- {messages}-----------")
    m_as_string = "\n\n".join([message.content for message in messages])
    qa_chain = (
        {
            "context": docsearch.vectorstore.as_retriever(),
            "summary": RunnableLambda(lambda _: summary),
            "conversation": RunnableLambda(lambda _: m_as_string),
            "question": RunnableLambda(lambda _: messages[-1].content)
        }
        | prompt
        | model
        # | StrOutputParser()
    )
    
    # print(f"------- doc search {docsearch.vectorstore.as_retriever()} ----------")
    response = await qa_chain.ainvoke(messages[-1].content)
    # response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
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

def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)

def format_is_relevant_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"{safety.unsafe_response}"
    )
    return AIMessage(content=content)

async def llama_guard_input(state: State, config: RunnableConfig) -> State:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}

async def llama_guard_output(state: State, config: RunnableConfig) -> State:
    # llama_guard = LlamaGuard()
    # safety_output = await llama_guard.ainvoke("User", state["messages"])
    # return {"output_safety": safety_output}
    checking_template = """You are an expert to monitoring the input question whether it relevant to mathematics and programming or not, if irrelevant to mathematics and programming, it will be marked as unsafe, other wise marked as safe:
    Response: {request}
    Answer:
    - First line JUST RETURN 'safe' word and 'unsafe' word in lowercase without adding any words.
    - If unsafe or irrelevant to mathematics and programming, a second line must include a response to sorry user and tell the reason naturally
    - If safe, a second line must include the reason why it safe"""
    prompt = ChatPromptTemplate.from_template(checking_template)
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    checking_output = prompt | model
    messages = state["messages"]
    response = await checking_output.ainvoke({"request": messages[-1].content})
    llama_guard = parse_llama_guard_relevant_topic(response.content)
    return {"is_relevant": llama_guard}
    
async def block_unsafe_content(state: State, config: RunnableConfig) -> State:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}

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

# Check for unsafe input and block further processing if found
def check_safety(state: State) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


# Define a new graph
workflow = StateGraph(State)

# Define the conversation node and the summarize node
workflow.add_node("conversation", acall_model)
# workflow.add_node("normal_conversation", normal_conversation)
workflow.add_node(summarize_conversation)
workflow.add_node("guard_input", llama_guard_input)
workflow.add_node("guard_output", llama_guard_output)
workflow.add_node("block_unsafe_content", block_unsafe_content)
workflow.add_node("block_unsafe_output_content", block_irrelevant_content)

# Set the entrypoint as conversation
workflow.set_entry_point("guard_input")

# workflow.add_edge("guard_input", "conversation")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "conversation",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

workflow.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "guard_output"}
)

workflow.add_conditional_edges(
    "guard_output", check_is_relevant, {"irrelevant": "block_unsafe_output_content", "relevant": "conversation" }
)

# We now add a normal edge from `summarize_conversation` to END.
# This means that after `summarize_conversation` is called, we end.
workflow.add_edge("summarize_conversation", END)
# Always END after blocking unsafe content
workflow.add_edge("block_unsafe_content", END)
workflow.add_edge("block_unsafe_output_content", END)

global_chatbot = workflow.compile()