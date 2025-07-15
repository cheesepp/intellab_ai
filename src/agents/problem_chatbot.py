import os
from langchain_openai import ChatOpenAI
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
from agents.global_chatbot import format_is_relevant_message, format_safety_message
from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment, parse_llama_guard_relevant_topic
from core import settings
from core.llm import get_model
from core.prompt_manager import PromptManager

prompt_manager = PromptManager(development_mode=True)
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

class AgentState(MessagesState):
    problem: str
    question: str
    source_code: str
    summary: str
    safety: LlamaGuardOutput
    is_relevant: LlamaGuardOutput
    
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
  
  # llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
  # llm = ChatOllama(model="llama3.2", base_url=OLLAMA_HOST)
  llm = ChatOpenAI(model="openai/gpt-4o-mini", base_url=os.getenv("OPENROUTER_BASE"))
  
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
  print(f"=========== SOLVER CHATBOT {summary} ===========")
  if summary:
    system_message = f"Summary of conversation earlier: {summary}"
    messages = [SystemMessage(content=system_message)] + state["messages"]
  else:
    messages = state["messages"]
  m_as_string = "\n\n".join([message.content for message in messages])     
  solver_model = {
        "conversation": RunnableLambda(lambda _: m_as_string),
        # "summary": RunnableLambda(lambda _: summary),
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
  code = state["source_code"]
  summary = state.get("summary", "")
  if summary:
    system_message = f"Summary of conversation earlier: {summary}"
    messages = [SystemMessage(content=system_message)] + state["messages"]
  else:
    messages = state["messages"]
  m_as_string = "\n\n".join([message.content for message in messages])    
  reviewer_model = {
        "conversation": RunnableLambda(lambda _: m_as_string),
        "problem": RunnableLambda(lambda _: problem),
        "source_code": RunnableLambda(lambda _: code),
        "question": RunnableLambda(lambda _: question),
      } | reviewer_prompt | llm
  
  response = await reviewer_model.ainvoke(messages)
  
  return {"messages": [response]}
  
def extract_message(state: AgentState):
    message = state["messages"][-1].content
    print (f"========= GO {message} ============")
    pattern = r"Problem:\s*(.*?)\s*Problem_id:\s*(\S+)\s*Code:\s*(.*?)\s*Question:\s*(.+)"

    match = re.search(pattern, message, re.DOTALL)
    print (f"========= MATCH {match} ============")
    if match:
        problem_content = match.group(1)
        problem_id = match.group(2)
        code = match.group(3)
        question = match.group(4)
        print("Problem Content:", problem_content)
        print("Problem ID:", problem_id)
        print("Source code:", code)
        print("Question:", question)
        return {"problem": problem_content, "question": question, "source_code": code}
    else:
        return {"problem": "", "question": ""}
  
def get_question(state: AgentState):
   if not state["question"]:
      return {"question": "What is the solution to this problem?"}
   return {"question": state["question"]}

async def acall_model(state: AgentState, config: RunnableConfig):
    problem = state["problem"]
    question = state["question"]
    
    print(f"======== {question, prompt} ==========")
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
    if len(messages) > 10:
        return "summarize_conversation"
    # Otherwise we can just end
    return END

async def summarize_conversation(state: AgentState, config: RunnableConfig):
    print(f"------- SUMMARIZE CONVERSATION ----------")
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    # model = ChatOllama(model="llama3.2", base_url=OLLAMA_HOST)
    # First, we summarize the conversation
    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            """
            Keep the summary clear, informative, and focused on the technical content.
            """
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    summary_messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(summary_messages)
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-4]]
    return {"summary": response.content, "messages": delete_messages}

async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}

# async def llama_guard_output(state: AgentState, config: RunnableConfig) -> AgentState:
#     # Simple keyword-based check for obviously irrelevant content
#     message_content = state["question"]
    
#     # List of clearly non-programming topics
#     irrelevant_keywords = [
#         "weather", "cooking", "recipe", "politics", "sports", "music", "movie", 
#         "restaurant", "travel", "fashion", "health", "medicine", "dating", 
#         "relationship", "news", "celebrity", "gossip", "shopping", "food",
#         "car", "vehicle", "real estate", "investment", "stock market", "sleep"
#     ]
    
#     # Check if message contains clearly irrelevant keywords
#     is_clearly_irrelevant = any(keyword in message_content for keyword in irrelevant_keywords)
    
#     if is_clearly_irrelevant and not any(prog_word in message_content for prog_word in 
#                                         ["code", "program", "algorithm", "software", "app", "tech", "computer"]):
#         # Only block if it's clearly irrelevant AND doesn't contain any programming context
#         guard_output = LlamaGuardOutput(
#             safety_assessment=SafetyAssessment.UNSAFE,
#             unsafe_response="I'd love to chat, but I'm here to help with programming and computer science topics! Got any coding questions I can help with?"
#         )
#         return {"is_relevant": guard_output}
#     else:
#         # Let everything else pass through (be more permissive)
#         guard_output = LlamaGuardOutput(
#             safety_assessment=SafetyAssessment.SAFE,
#             unsafe_response=""
#         )
#         return {"is_relevant": guard_output}
async def llama_guard_output(state: AgentState, config: RunnableConfig) -> AgentState:
    # llama_guard = LlamaGuard()
    # safety_output = await llama_guard.ainvoke("User", state["messages"])
    # return {"output_safety": safety_output}
    checking_template = """You are an expert at determining whether an input is relevant to programming or casual conversation that should be allowed in a programming-focused chatbot.

Analyze the 'Latest User Input' based on the 'Conversation Summary' and 'Recent Conversation History' to understand the context.

Conversation Summary:
{summary}

Recent Conversation History:
{conversation_history}

Latest User Input: {question}

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

    Input: "Hi, who are you?"
    Output:
    relevant
    This is a casual greeting that is appropriate for a programming-focused chatbot.

    Input: "Can you introduce yourself?"
    Output:
    relevant
    This is a casual conversation that is appropriate for a programming-focused chatbot.

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
    summary = state.get("summary", "No summary yet.")
    conversation_history = "\n\n".join(
        [
            f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
            for m in messages
        ]
    )
    question = state["question"]
    response = await checking_output.ainvoke({
        "summary": summary,
        "conversation_history": conversation_history,
        "question": question
    })
    llama_guard = parse_llama_guard_relevant_topic(response.content)
    return {"is_relevant": llama_guard}
    
async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}

async def block_irrelevant_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["is_relevant"]
    return {"messages": [format_is_relevant_message(safety)]}


# Check for unsafe input and block further processing if found
def check_is_relevant(state: AgentState) -> Literal["relevant", "irrelevant"]:
    is_relevant: LlamaGuardOutput = state["is_relevant"]
    match is_relevant.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "irrelevant"
        case _:
            return "relevant"

# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


# Define the graph
agent = StateGraph(MessagesState)
agent.add_node("model", acall_model)
agent.add_node("guard_input", llama_guard_input)
agent.add_node("guard_output", llama_guard_output)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("block_unsafe_output_content", block_irrelevant_content)
agent.add_node("get_question", get_question)
agent.add_node("solver", solver)
agent.add_node("reviewer", reviewer)
agent.add_node("extract", extract_message)
agent.add_node(summarize_conversation)
agent.set_entry_point("extract")

agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "guard_output"}
)

agent.add_conditional_edges(
    "guard_output", check_is_relevant, {"irrelevant": "block_unsafe_output_content", "relevant": "get_question" }
)

agent.add_edge(
  "extract",
  "guard_input"
)

agent.add_conditional_edges(
  "get_question",
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
