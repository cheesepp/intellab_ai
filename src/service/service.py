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

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status, Request
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

from .routers.history import router as history_router
from .routers.streaming_agents import router as streaming_router
from .routers.invoking_agents import router as invoking_router


warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


# @asynccontextmanager
# async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
#     # Construct agent with Sqlite checkpointer
#     # TODO: It's probably dangerous to share the same checkpointer on multiple agents
#     async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
#         agents = get_all_agent_info()
#         for a in agents:
#             agent = get_agent(a.key)
#             agent.checkpointer = saver
#         yield
#     # context manager will clean up the AsyncSqliteSaver on exit

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Construct agent with Sqlite checkpointer
    # TODO: It's probably dangerous to share the same checkpointer on multiple agents
    async with AsyncConnectionPool(
        # Example configuration
        conninfo=os.getenv("DB_CONNECTION_STRING"),
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        # NOTE: you need to call .setup() the first time you're using your checkpointer
        await checkpointer.setup()
        agents = get_all_agent_info()
        for a in agents:
            agent = get_agent(a.key)
            agent.checkpointer = checkpointer
        yield
    # context manager will clean up the AsyncSqliteSaver on exit


app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/ai", dependencies=[Depends(verify_bearer)])
router.include_router(streaming_router)
router.include_router(invoking_router)
router.include_router(history_router)

@router.get("/info")
async def info(request: Request) -> ServiceMetadata:
    # Extract the X-UserRole header
    user_role_header = request.headers.get("X-UserRole", "")
    print(f"User Role Header: {user_role_header}")
    user_role, premium_plan = user_role_header.split(",") if user_role_header else ("", "")

    # Log the extracted values
    print(f"User Role: {user_role}, Premium Plan: {premium_plan}")

    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@router.get("/pdf-summary", tags=["Summarize Agent"])
async def get_pdf():
    pdf_path = os.path.join(os.getcwd(), f"summary.pdf")

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    response = FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"summarization.pdf"
    )

    # Delete the file after sending it
    # os.remove(pdf_path)
    return response

# from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.include_router(router)
