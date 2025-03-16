import os
from langgraph.graph import END, MessagesState, StateGraph
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnableConfig,
)
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
import re
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from core import settings
from core.llm import get_model
from core.prompt_manager import PromptManager

prompt_manager = PromptManager(development_mode=True)
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

class AgentState(MessagesState):
    problem: str
    question: str
    summary: str
    
class CheckRequireSolving(TypedDict):
  """"""
prompt = ChatPromptTemplate.from_template(prompt_manager.PROBLEM_ADVISOR_TEMPLATE)

# Data model
class CheckSolveProblem(BaseModel):
    """Binary score to check whether user want to solve problem."""

    want_to_solve: str = Field(
        description="The answer of user want to solve problem, 'yes' or 'no'"
    )

# Data model
class CheckReviewProblem(BaseModel):
    """Binary score to check whether user want to review the solution."""

    want_to_review: str = Field(
        description="The answer of user want to review the solution, 'yes' or 'no'"
    )
    
# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant expectation."""

    cases: Literal["normal_conversation", "want_to_solve", "want_to_review"] = Field(
        ...,
        description="Given a user question choose to route it to normal conversation, problem solving or solution reviewing.",
    )

# ROUTE_QUERY_TEMPLATE = """You are an expert at routing a user question to normal conversation agent, problem-solving agent or solution-reviewing agent.
# Based on user's question to determine whether user want to solve the problem or want to review or want to have a normal chat. 
# Give an answer based on each case:
# - 'normal_conversation': return when user want to have a normal chat.
# - 'want_to_solve': return when user want to solve the problem.
# - 'want_to_review': return when user want to review about code or solution.

# Question: {question}
# """

async def route_query(state: AgentState, config:RunnableConfig) -> Literal["model", "solver", "reviewer"]:
  
#   llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
  llm = ChatOllama(model="llama3.2", base_url=OLLAMA_HOST)
  
  ROUTE_QUERY_PROMPT = ChatPromptTemplate.from_template(prompt_manager.PROBLEM_ROUTE_TEMPLATE)
  
  prob_checker = ROUTE_QUERY_PROMPT | llm.with_structured_output(RouteQuery)
  
  response: RouteQuery = await prob_checker.ainvoke({"question": state["question"]})
  # prob_checker2 = CHECK_SOLVE_PROMPT | llm
  
  # response2 = await prob_checker2.ainvoke({"question": state["question"]})
  
  # print(response2.content, response)
  if response.cases == 'want_to_solve':
    print("=========== WANT TO SOLVE ===========")
    return "solver"
  if response.cases == 'want_to_review':
    print("=========== WANT TO REVIEW ===========")
    return "reviewer"
  print("=========== NORMAL CONVERSATION ===========")
  return "model"

async def solver(state: AgentState, config: RunnableConfig):
  # solver_template = """You are an expert in problem-solving for programming. Your mission is to guide the user on how to solve the given problem without recommending a solution. Provide hints and a professional approach to tackling the problem, but do not give the solution.
  
  # Problem: {problem}
  # """
  llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
#   llm = ChatOllama(model="codeqwen", base_url=OLLAMA_HOST)
  solver_prompt = ChatPromptTemplate.from_template(prompt_manager.PROBLEM_SOLVER_TEMPLATE)
  problem = state["problem"]
  question = state["question"]
  summary = state.get("summary","")
  if summary:
    system_message = f"Summary of conversation earlier: {summary}"
    messages = [SystemMessage(content=system_message)] + state["messages"]
  else:
    messages = state["messages"]
        
  solver_model = {
        "problem": RunnableLambda(lambda _: problem),
        "question": RunnableLambda(lambda _: question),
      } | solver_prompt | llm
  response = await solver_model.ainvoke(messages)
  return {"messages": [response]}

async def reviewer(state: AgentState, config: RunnableConfig):
  
  # reviewer_template = """You are an expert in reviewing code. Your mission is analyze complexity, how to optimize to improve performance and reformat following style guides from the given problem and solution. Give feedback to code naturally, meaningful and understandable.
  # Problem: {problem}
  # User's solution: {question}
  # Answer includes these information:
  # - Time complexity from the given solution:
  # - Space complexity from the given solution:
  # - Give feedback
  # - Suggest better code"""
  llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
#   llm = ChatOllama(model="codeqwen", base_url=OLLAMA_HOST)
  
  reviewer_prompt = ChatPromptTemplate.from_template(prompt_manager.PROBLEM_REVIEWER_TEMPLATE)
  
  problem = state["problem"]
  question = state["question"]
  summary = state.get("summary", "")
  if summary:
    system_message = f"Summary of conversation earlier: {summary}"
    messages = [SystemMessage(content=system_message)] + state["messages"]
  else:
    messages = state["messages"]
        
  reviewer_model = {
        "problem": RunnableLambda(lambda _: problem),
        "question": RunnableLambda(lambda _: question),
      } | reviewer_prompt | llm
  
  response = await reviewer_model.ainvoke(messages)
  
  return {"messages": [response]}
  
def extract_message(state: AgentState):
    message = state["messages"][-1].content
    print (f"========= GO {message} ============")
    pattern = r"Problem:\s*(.*?)\s*Problem_id:\s*(\S+)\s*Question:\s*(.+)"

    match = re.search(pattern, message, re.DOTALL)
    print (f"========= MATCH {match} ============")
    if match:
        problem_content = match.group(1)
        problem_id = match.group(2)
        question = match.group(3)
        print("Problem Content:", problem_content)
        print("Problem ID:", problem_id)
        print("Question:", question)
        return {"problem": problem_content, "question": question}
    else:
        return {"problem": "", "question": ""}
   
async def acall_model(state: AgentState, config: RunnableConfig):
    problem = state["problem"]
    question = state["question"]
    
    print(f"======== {question} ==========")
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
        "problem": RunnableLambda(lambda _: problem),
        "question": RunnableLambda(lambda _: question),
      } | prompt | model
    response = await normal_chatbot.ainvoke(messages)
    return {"messages": [response]}

# We now define the logic for determining whether to end or summarize the conversation
def should_continue(state: AgentState) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    # Otherwise we can just end
    return END

async def summarize_conversation(state: AgentState, config: RunnableConfig):
    print(f"------- SUMMARIZE CONVERSATION ----------")
    # model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model = ChatOllama(model="llama3.2", base_url=OLLAMA_HOST)
    # First, we summarize the conversation
    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            """
            Focus on:
            1. The main programming concepts discussed
            2. Key advice or strategies provided
            3. Important questions raised
            4. Any conclusions or next steps

            Keep the summary clear, informative, and focused on the technical content.
            """
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    summary_messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(summary_messages)
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
  
# Define the graph
agent = StateGraph(MessagesState)
agent.add_node("model", acall_model)
agent.add_node("solver", solver)
agent.add_node("reviewer", reviewer)
agent.add_node("extract", extract_message)
agent.add_node(summarize_conversation)
agent.set_entry_point("extract")

agent.add_conditional_edges(
  "extract",
  route_query
)

# We now add a conditional edge
agent.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "model",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a conditional edge
agent.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "solver",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a conditional edge
agent.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "reviewer",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

agent.add_edge("summarize_conversation", END)

problem_chatbot = agent.compile()
