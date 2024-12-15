from langchain_core.messages import BaseMessage
from typing import Annotated, TypedDict
from langchain_core.documents import Document
from langgraph.graph import END, add_messages


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    candidate_answer: str
    retries: int
    web_fallback: bool