import datetime
import json
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

from schema import ChatMessage
from schema.schema import UserInput
from core.database import collection
from uuid import UUID, uuid4

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
    if user_id:
        print(f'CONTAIN USER ID {ai_output.metadata}')
        # Create message entries
        user_entry = {"type": "user", "content": human_message, "timestamp": timestamp}
        ai_entry = {
            "type": "ai",
            "content": ai_message,
            "metadata": {'model': ai_output.metadata["model"]},
            "timestamp": ai_output.metadata["created_at"]
        }
        # Check if the user exists
        user_data = collection.find_one({"user_id": user_id})
        if user_data:
            if thread_id:
                print("CONTAIN conversation_id")
                # Check if conversation exists
                conversation = collection.find_one(
                    {"user_id": user_id, "conversations.thread_id": thread_id}
                )
                
                if conversation:
                    # Append new messages to the existing conversation
                    collection.update_one(
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

                    collection.update_one(
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

            collection.insert_one(new_user)
            print({"message": "New user created and conversation started", "thread_id": thread_id})
        
    else:
        raise HTTPException(status_code=400, detail="Invalid user id!")

async def store_title(user_input: UserInput, output: str, thread_id: str):
    user_id = user_input.user_id
    thread_id = thread_id
    conversation = collection.find_one({"user_id": user_id, "conversations.thread_id": thread_id})
    if not user_id or not thread_id:
        raise HTTPException(status_code=400, detail="User ID and thread ID are required")

    # Check if conversation exists
    conversation = collection.find_one(
        {"user_id": user_id, "conversations.thread_id": thread_id}
    )

    if conversation:
        # Update the specific conversation within the conversations array
        collection.update_one(
            {"user_id": user_id, "conversations.thread_id": thread_id},
            {"$set": {"conversations.$.title": output}}
        )
        print({"message": "Summary title added successfully", "thread_id": thread_id})
        return
    raise HTTPException(status_code=404, detail="Conversation not found")
        
        
    