from datetime import datetime
import json
import logging
import os
import re
import warnings
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from agents import DEFAULT_AGENT, get_agent, get_all_agent_info
from agents.summarize_agent import extract_course_info
from core import settings
from schema import (
    StreamInput,
)
from service.utils import (
    _parse_input,
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
    store_chat_history,
    store_problem_chat_history,
    store_title,
    store_problem_title
)
from agents.global_chatbot import workflow
from fpdf import FPDF

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
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
    # Process streamed events from the graph and yield messages over the SSE stream.
    async for event in agent.astream_events(**kwargs, version="v2"):
        if not event:
            continue
        # Check if event["metadata"] is a string and convert it to a dictionary
        metadata_dict = event["metadata"]
        # print("================ DATA ===============")
        # print(event)
        if "langgraph_node" in metadata_dict and metadata_dict["langgraph_node"] == "guard_output":
            print("================ GUARD OUTPUT ===============")
            print(event)
        if "langgraph_node" in metadata_dict and metadata_dict["langgraph_node"] == "summarize_conversation":
            continue
        new_messages = []
        # Yield messages written to the graph state after node execution finishes.
        if (
            event["event"] == "on_chain_end"
            # on_chain_end gets called a bunch of times in a graph execution
            # This filters out everything except for "graph node finished"
            and any(t.startswith("graph:step:") for t in event.get("tags", []))
            and "messages" in event["data"]["output"]
        ):
            new_messages = event["data"]["output"]["messages"]
        # Also yield intermediate messages from agents.utils.CustomData.adispatch().
        if event["event"] == "on_custom_event" and "custom_data_dispatch" in event.get("tags", []):
            new_messages = [event["data"]]

        for message in new_messages:
            try:
                chat_message = langchain_to_chat_message(message)
                chat_message.run_id = str(run_id)
                chat_message.thread_id = str(thread_id)
            except Exception as e:
                logger.error(f"Error parsing message: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                continue
            # LangGraph re-sends the input message, which feels weird, so drop it
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            if agent_id == "global_chatbot":
                print("STORE CHAT")
                await store_chat_history(user_input, chat_message, thread_id, timestamp)
            if agent_id == "problem_chatbot":
                print("STORE PROBLEM CHAT")
                await store_problem_chat_history(user_input, chat_message, thread_id, timestamp)
            elif agent_id == "title_generator":
                if user_input.problem_title == "1":
                    await store_problem_title(user_input, chat_message.content, thread_id)
                else:
                    await store_title(user_input, chat_message.content, thread_id)
                    
            elif agent_id == "summarize_assistant":
            
                extract_values = extract_course_info(user_input.message)
                pdf_path = os.path.join(os.getcwd(), "summary.pdf")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f'{extract_values["course_name"]} Summary', ln=True, align='C')
                pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='L')
                pdf.multi_cell(0, 10, txt=chat_message.content)  # Add output content to the PDF
                print(f"Extract value {extract_values}")
                # Write PDF to file
                pdf.output(pdf_path)
            print(message)
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

        # Yield tokens streamed from LLMs.
        if (
            event["event"] == "on_chat_model_stream"
            and user_input.stream_tokens
            and "llama_guard" not in event.get("tags", [])
        ):
            content = remove_tool_calls(event["data"]["chunk"].content)
            if content:
                # Empty content in the context of OpenAI usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content.
                yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
            continue

    yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }

router = APIRouter(prefix="/stream")

@router.post("/summarize_agent",  tags=["Summarize Agent"], 
             response_class=StreamingResponse, responses=_sse_response_example(),
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
             response_class=StreamingResponse, responses=_sse_response_example(),
             description="This agent is a general chatbot.")
@router.post("/problem_chatbot",  tags=["Problem Chatbot"], 
             response_class=StreamingResponse, responses=_sse_response_example(),
             description="""
             This agent is problem chatbot used in problem detail page.
             Request format:
             
             {
                message: "Problem: <problem> Problem_id: <problem_id> Question: <question>"
                user_id:
                thread_id:
                model: 
             }
             """)
@router.post("/title_generator",  tags=["Title Generator"],
             response_class=StreamingResponse, responses=_sse_response_example(),
             description="""
             This agent used to generate title for each conversation
             Request format:
             
             {
                message: <First user's message of conversation>
                user_id:
                thread_id:
                model: 
             }
             if it used to generate title for conversation in problem, please add these params:
             {
                 "problem_title": "1",
                 "problem_id": "123" 
             }
             """)
@router.post("/invoke",  tags=["Invoke Default Agent"], 
             response_class=StreamingResponse, responses=_sse_response_example(),
            description="""
            Stream an agent's response to a user input, including intermediate messages and tokens.

            If agent_id is not provided, the default agent will be used.
            Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
            is also attached to all messages for recording feedback.

            Set `stream_tokens=false` to return intermediate messages but not token-by-token.
            """)
# @router.post(
#     "/{agent_id}", response_class=StreamingResponse, responses=_sse_response_example(),
#     tags=["Streaming Agents"]
# )
# @router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(request: Request, user_input: StreamInput) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    agent_id = request.url.path.split("/")[-1]  
    if (agent_id == "invoke"):
        agent_id = DEFAULT_AGENT
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )