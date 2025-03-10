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

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
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
from core.database import collection

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/invoke")

@router.post("/summarize_agent",  tags=["Summarize Agent"], 
             description="""
             This agent summarizes lessons from a course.
             Request format:
             
             {
                message: "course name: <course_name>, id: <course_id>, regenerate: <true/false as string type>
                user_id:
                thread_id:
                model: 
             }
             """)
@router.post("/global_chatbot",  tags=["Global Chatbot"], 
             description="This agent is a general chatbot.")
@router.post("/problem_chatbot",  tags=["Problem Chatbot"], 
             description="""
             This agent is problem chatbot used in problem detail page.
             Request format:
             
             {
                message: "Problem: <problem> Question: <question>"
                user_id:
                thread_id:
                model: 
             }
             """)
@router.post("/title_generator",  tags=["Title Generator"],
             description="""
             This agent used to generate title for each conversation
             Request format:
             
             {
                message: <First user's message of conversation>
                user_id:
                thread_id:
                model: 
             }
             """)
@router.post("/invoke",  tags=["Invoke Default Agent"], 
            description="""
            Invoke an agent with user input to retrieve a final response.
            
            If agent_id is not provided, the default agent will be used.
            Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
            is also attached to messages for recording feedback.
            """)
async def invoke(request: Request, user_input: UserInput) -> ChatMessage:

    agent_id = request.url.path.split("/")[-1]  
    if (agent_id == "invoke"):
        agent_id = DEFAULT_AGENT
    agent: CompiledStateGraph = get_agent(agent_id)
       
    kwargs, run_id, thread_id = _parse_input(user_input)
    timestamp = datetime.now().isoformat()
    if agent_id == "title_generator":
        fake_thread_id = uuid4()
        run_id = uuid4()
        kwargs = {
        "input": {"messages": [HumanMessage(content=user_input.message)]},
        "config": RunnableConfig(
            configurable={"thread_id": str(fake_thread_id), "model": user_input.model}, run_id=run_id
        ),
    }
    try:
        response = await agent.ainvoke(**kwargs)
        print(response)
        output = langchain_to_chat_message(response["messages"][-1])
        output.run_id = str(run_id)
        output.thread_id = str(thread_id)
        if agent_id == "global_chatbot":
            print("STORE CHAT")
            await store_chat_history(user_input, output, thread_id, timestamp)
        elif agent_id == "title_generator":
            await store_title(user_input, output.content, thread_id)
        # Generate and store a PDF if agent_id is 'summarize-assistant'
        elif agent_id == "summarize_assistant":
            
            extract_values = extract_course_info(user_input.message)
            pdf_path = os.path.join(os.getcwd(), "summary.pdf")
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f'{extract_values["course_name"]} Summary', ln=True, align='C')
            pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='L')
            pdf.multi_cell(0, 10, txt=output.content)  # Add output content to the PDF
            print(f"Extract value {extract_values}")
            # Write PDF to file
            pdf.output(pdf_path)
            
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")