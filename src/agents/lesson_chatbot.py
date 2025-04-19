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

When responding, first identify the specific sentence/concept in question, then provide a thorough explanation that places it within the broader context of the lesson material."""

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_HOST
)
# Create an index using the loaded documents
prompt = ChatPromptTemplate.from_template(template)

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
        print("Problem Content:", lesson_name)
        print("Problem ID:", lesson_id)
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
agent.add_node(summarize_conversation)
agent.set_entry_point("extract")

agent.add_edge("extract", "retrieve")
agent.add_edge("retrieve", "model")
agent.add_edge("model", "summarize_conversation")
agent.add_edge("summarize_conversation", END)


lesson_chatbot = agent.compile()