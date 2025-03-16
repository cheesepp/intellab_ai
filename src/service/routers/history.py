from datetime import datetime
import json
import logging
import os
import re
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, List
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client as LangsmithClient
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agents import DEFAULT_AGENT, get_agent, get_all_agent_info
from agents.summarize_agent import extract_course_info
from core import settings
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    _parse_input,
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
    store_chat_history,
    store_title
)
from agents.global_chatbot import workflow
from fpdf import FPDF
from io import BytesIO
import tempfile
from core.database import global_chatbot_collection
from core.database import problem_chatbot_collection

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversation")

# @router.post("/history")
# def history(input: ChatHistoryInput) -> ChatHistory:
#     """
#     Get chat history.
#     """
#     # TODO: Hard-coding DEFAULT_AGENT here is wonky
#     agent: CompiledStateGraph = get_agent(DEFAULT_AGENT)
#     try:
#         state_snapshot = agent.get_state(
#             config=RunnableConfig(
#                 configurable={
#                     "thread_id": input.thread_id,
#                 }
#             )
#         )
#         messages: list[AnyMessage] = state_snapshot.values["messages"]
#         chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
#         return ChatHistory(messages=chat_messages)
#     except Exception as e:
#         logger.error(f"An exception occurred: {e}")
#         raise HTTPException(status_code=500, detail="Unexpected error")

@router.get("/{user_id}/threads", response_model=dict, tags=["History"])
def get_all_thread_ids(user_id: str):
    conversation = global_chatbot_collection.find_one({"user_id": user_id}, {"_id": 0, "conversations": 1})
    if not conversation or "conversations" not in conversation:
        return {"code": 404, "status": "No conversations found for the user", "data": []}
    threads = [{"thread_id": convo.get("thread_id"), "title": convo.get("title", "No title available"), "timestamp": convo.get("timestamp")} for convo in conversation["conversations"] if "thread_id" in convo]
    return {"code": 200, "status": "Success", "data": threads}

@router.get("/{user_id}/thread/{thread_id}", response_model=dict, tags=["History"])
def get_conversation_by_user_and_thread(user_id: str, thread_id: str):
    conversation = global_chatbot_collection.find_one({"user_id": user_id, "conversations.thread_id": thread_id}, {"_id": 0, "conversations.$": 1})
    if not conversation:
        return {"code": 404, "status": "Thread not found for the given user", "data": {}}
    return {"code": 200, "status": "Success", "data": conversation["conversations"][0]}

@router.get("/{user_id}/problem/{problem_id}/threads", response_model=dict, tags=["History"])
def get_all_thread_ids(user_id: str, problem_id: str):
    conversation = problem_chatbot_collection.find_one(
        {"user_id": user_id},
        {"_id": 0, "conversations": 1}
    )

    if not conversation or "conversations" not in conversation:
        return {"code": 404, "status": "No conversations found for the user", "data": []}
    
    # Filter conversations that match the problem_id
    threads = [
        {
            "thread_id": convo.get("thread_id"),
            "title": convo.get("title", "No title available"),
            "timestamp": convo.get("timestamp")
        }
        for convo in conversation["conversations"]
        if "thread_id" in convo and convo.get("problem_id") == problem_id  # Filter by problem_id
    ]

    if not threads:
        return {"code": 404, "status": "No conversations found for this problem", "data": []}

    return {"code": 200, "status": "Success", "data": threads}

@router.get("/{user_id}/problem/{problem_id}/thread/{thread_id}", response_model=dict, tags=["History"])
def get_conversation_by_user_problem_and_thread(user_id: str, problem_id: str, thread_id: str):
    conversation = problem_chatbot_collection.find_one(
        {"user_id": user_id, "conversations.thread_id": thread_id, "conversations.problem_id": problem_id},
        {"_id": 0, "conversations.$": 1}
    )
    
    if not conversation:
        return {"code": 404, "status": "Thread not found for the given user and problem", "data": {}}
    
    return {"code": 200, "status": "Success", "data": conversation["conversations"][0]}
