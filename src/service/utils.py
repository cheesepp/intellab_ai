from datetime import datetime
import json
import re
from fastapi import HTTPException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)

from langchain_core.runnables import RunnableConfig
from schema import ChatMessage
from schema.schema import UserInput
from core.database import global_chatbot_collection, problem_chatbot_collection
from uuid import UUID, uuid4
from typing import Annotated, Any, List

def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], UUID, UUID]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    kwargs = {
        "input": {"messages": [HumanMessage(content=user_input.message)]},
        "config": RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model}, run_id=run_id
        ),
    }
    return kwargs, run_id, thread_id

def extract_course_info(input_string):
    # Regular expression to match the course name, ID, and regenerate flag
    pattern = r"course name: (.*?), id: (.*?), regenerate: (true|false)"
    
    # Search the string for matches
    match = re.search(pattern, input_string, re.IGNORECASE)
    
    if match:
        # Extracted groups
        course_name = match.group(1)
        course_id = match.group(2)
        regenerate = match.group(3).lower() == 'true'  # Convert to boolean
        return {
            "course_name": course_name,
            "course_id": course_id,
            "regenerate": regenerate
        }
    else:
        raise ValueError("Input string does not match the expected format.")
    
def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """Create a ChatMessage from a LangChain message."""
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.metadata = message.response_metadata
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            if message.role == "custom":
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=message.content[0],
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]

async def store_chat_history(user_input: UserInput, ai_output: ChatMessage, thread_id: UUID, timestamp: str):
    user_id = user_input.user_id
    thread_id = thread_id
    human_message = user_input.message
    ai_message = ai_output.content
    ai_timestamp = datetime.now().isoformat()
    if user_id:
        print(f'CONTAIN USER ID {ai_output.metadata}')
        # Create message entries
        user_entry = {"type": "user", "content": human_message, "timestamp": timestamp}
        ai_entry = {
            "type": "ai",
            "content": ai_message,
            "metadata": {'model': user_input.model},
            "timestamp": ai_timestamp
        }
        # Check if the user exists
        user_data = global_chatbot_collection.find_one({"user_id": user_id})
        if user_data:
            if thread_id:
                print("CONTAIN conversation_id")
                # Check if conversation exists
                conversation = global_chatbot_collection.find_one(
                    {"user_id": user_id, "conversations.thread_id": thread_id}
                )
                
                if conversation:
                    # Append new messages to the existing conversation
                    global_chatbot_collection.update_one(
                        {"user_id": user_id, "conversations.thread_id": thread_id},
                        {"$push": {"conversations.$.messages": {"$each": [user_entry, ai_entry]}}}
                    )
                    print({"message": "Conversation updated", "thread_id": thread_id})
                else:
                    print("GENERATE NEW CONVERSATION")
                    # Create a new conversation if no conversation_id is provided
                    # conversation_id = f"{user_id}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                    new_conversation = {
                        "thread_id": thread_id,
                        "timestamp": timestamp,
                        "messages": [user_entry, ai_entry]
                    }

                    global_chatbot_collection.update_one(
                        {"user_id": user_id},
                        {"$push": {"conversations": new_conversation}},
                        upsert=True
                    )
                    print({"message": "New conversation created", "thread_id": thread_id})   
            else:
                raise HTTPException(status_code=500, detail="Internal Server Error")
        
        else:
            print("ADD NEW USER TO DATABASE")
            # User does not exist -> Create a new user document
            # conversation_id = f"{user_id}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            new_user = {
                "user_id": user_id,
                "conversations": [
                    {
                        "thread_id": thread_id,
                        "timestamp": timestamp,
                        "messages": [user_entry, ai_entry]
                    }
                ]
            }

            global_chatbot_collection.insert_one(new_user)
            print({"message": "New user created and conversation started", "thread_id": thread_id})
        
    else:
        raise HTTPException(status_code=400, detail="Invalid user id!")

def extract_message(message):
        print (f"========= {message} ============")
        pattern = r"Problem:\s*(.*?)\s*Problem_id:\s*(\S+)\s*Question:\s*(.+)"

        match = re.search(pattern, message, re.DOTALL)
        print (f"========= {match} ============")
        if match:
            problem_content = match.group(1)
            problem_id = match.group(2)
            question = match.group(3)
            print("Problem Content:", problem_content)
            print("Problem ID:", problem_id)
            print("Question:", question)
            return {"problem": problem_content, "problem_id": problem_id, "question": question}
        else:
            return {"problem": "", "problem_id": "", "question": ""}
        
async def store_problem_chat_history(user_input: UserInput, ai_output: ChatMessage, thread_id: UUID, timestamp: str):
    
    user_id = user_input.user_id
    thread_id = thread_id
    extracted_message = extract_message(user_input.message)
    human_message = extracted_message["question"]
    problem_id = extracted_message["problem_id"]
    
    ai_message = ai_output.content
    ai_timestamp = datetime.now().isoformat()
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid user id!")

    print(f'CONTAIN USER ID {ai_output.metadata}')

    user_entry = {"type": "user", "content": human_message, "timestamp": timestamp}
    ai_entry = {
        "type": "ai",
        "content": ai_message,
        "metadata": {'model': user_input.model},
        "timestamp": ai_timestamp
    }

    user_data = problem_chatbot_collection.find_one({"user_id": user_id})

    if user_data:
        if thread_id:
            print("CONTAIN conversation_id")

            conversation = problem_chatbot_collection.find_one(
                {"user_id": user_id, "conversations.thread_id": thread_id, "conversations.problem_id": problem_id}
            )

            if conversation:
                problem_chatbot_collection.update_one(
                    {"user_id": user_id, "conversations.thread_id": thread_id, "conversations.problem_id": problem_id},
                    {"$push": {"conversations.$.messages": {"$each": [user_entry, ai_entry]}}}
                )
                print({"message": "Conversation updated", "thread_id": thread_id})
            else:
                print("GENERATE NEW CONVERSATION")
                
                # **Find all conversations with the same problem_id**
                problem_conversations = [
                    convo for convo in user_data["conversations"] if convo["problem_id"] == problem_id
                ]

                if len(problem_conversations) >= 3:
                    # **Find and remove the oldest conversation**
                    oldest_conversation = min(problem_conversations, key=lambda c: c["timestamp"])
                    problem_chatbot_collection.update_one(
                        {"user_id": user_id},
                        {"$pull": {"conversations": {"thread_id": oldest_conversation["thread_id"]}}}
                    )
                    print({"message": "Oldest conversation removed", "removed_thread_id": oldest_conversation["thread_id"]})

                # **Add the new conversation**
                new_conversation = {
                    "thread_id": thread_id,
                    "problem_id": problem_id,
                    "timestamp": timestamp,
                    "messages": [user_entry, ai_entry]
                }

                problem_chatbot_collection.update_one(
                    {"user_id": user_id},
                    {"$push": {"conversations": new_conversation}},
                    upsert=True
                )
                print({"message": "New conversation created", "thread_id": thread_id})   
        else:
            raise HTTPException(status_code=500, detail="Internal Server Error")

    else:
        print("ADD NEW USER TO DATABASE")
        
        new_user = {
            "user_id": user_id,
            "conversations": [
                {
                    "thread_id": thread_id,
                    "problem_id": problem_id,
                    "timestamp": timestamp,
                    "messages": [user_entry, ai_entry]
                }
            ]
        }

        problem_chatbot_collection.insert_one(new_user)
        print({"message": "New user created and conversation started", "thread_id": thread_id, "problem_id": problem_id})

async def store_title(user_input: UserInput, output: str, thread_id: str):
    user_id = user_input.user_id
    thread_id = thread_id
    conversation = global_chatbot_collection.find_one({"user_id": user_id, "conversations.thread_id": thread_id})
    if not user_id or not thread_id:
        raise HTTPException(status_code=400, detail="User ID and thread ID are required")

    # Check if conversation exists
    conversation = global_chatbot_collection.find_one(
        {"user_id": user_id, "conversations.thread_id": thread_id}
    )

    if conversation:
        # Update the specific conversation within the conversations array
        global_chatbot_collection.update_one(
            {"user_id": user_id, "conversations.thread_id": thread_id},
            {"$set": {"conversations.$.title": output}}
        )
        print({"message": "Summary title added successfully", "thread_id": thread_id})
        return
    raise HTTPException(status_code=404, detail="Conversation not found")

async def store_problem_title(user_input: UserInput, output: str, thread_id: str):
    problem_id = user_input.problem_id
    user_id = user_input.user_id

    if not user_id or not thread_id or not problem_id:
        raise HTTPException(status_code=400, detail="User ID, thread ID, and problem ID are required")

    # Check if the conversation exists for the given user, thread, and problem_id
    conversation = problem_chatbot_collection.find_one(
        {"user_id": user_id, "conversations": {"$elemMatch": {"thread_id": thread_id, "problem_id": problem_id}}}
    )

    if conversation:
        # Update the specific conversation's title within the conversations array
        problem_chatbot_collection.update_one(
            {"user_id": user_id, "conversations.thread_id": thread_id, "conversations.problem_id": problem_id},
            {"$set": {"conversations.$.title": output}}
        )
        print({"message": "Summary title added successfully", "thread_id": thread_id, "problem_id": problem_id})
        return

    raise HTTPException(status_code=404, detail="Conversation not found for the given problem_id")

        
        
    