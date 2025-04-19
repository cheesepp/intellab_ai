from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from agents.global_chatbot import global_chatbot
from agents.problem_chatbot import problem_chatbot
from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.lesson_chatbot import lesson_chatbot
from agents.research_assistant import research_assistant
from agents.summarize_agent import summarize_assistant
from agents.title_generator import title_generator
from schema import AgentInfo

DEFAULT_AGENT = "summarize_assistant"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "global_chatbot": Agent(description="A chatbot used in the entire website.", graph=global_chatbot),
    "problem_chatbot": Agent(description="A chatbot used for problem-solving.", graph=problem_chatbot),
    "lesson_chatbot": Agent(description="A chatbot used for lesson explanation.", graph=lesson_chatbot),
    "chatbot": Agent(description="A simple chatbot.", graph=chatbot),
    "research_assistant": Agent(
        description="A research assistant with web search and calculator.", graph=research_assistant
    ),
    "summarize_assistant": Agent(
        description="A summarize assistant by retrieval data from database.", graph=summarize_assistant
    ),
    "title_generator": Agent(
        description="A summary title assistant from human message.", graph=title_generator
    ),
    "bg_task_agent": Agent(description="A background task agent.", graph=bg_task_agent),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
