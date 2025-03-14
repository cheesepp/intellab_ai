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

import re
from fpdf import FPDF

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


# ==================== UTILS FOR GLOBAL CHATBOT ======================    
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



# ==================== UTILS FOR PROBLEM CHATBOT ======================    
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

# ==================== UTILS FOR TITLE GENERATOR AGENT ===================
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


# =============== UTILS FOR SUMMARIZE AGENT ================
def process_markdown(md_text, pdf):
    lines = md_text.split("\n")
    
    for line in lines:
        if not line.strip():  # Skip empty lines
            pdf.ln(5)
            continue
            
        # Process bullet points
        if line.strip().startswith('• '):
            bullet_content = line.strip()[2:]  # Remove the bullet marker
            
            # Check if new page is needed
            if pdf.get_y() > 270:
                pdf.add_page()
                
            # Set initial position for bullet
            current_x = pdf.l_margin
            pdf.set_x(current_x)
            
            # Add bullet point
            pdf.cell(5, 10, "•", 0, 0)
            current_x += 8  # Space after bullet
            pdf.set_x(current_x)
            
            # Process the bullet content with formatting
            process_formatted_text(bullet_content, pdf, current_x)
            pdf.ln(10)
            
        elif line.strip().startswith('Lesson: '):
            # Handle lesson text
            if pdf.get_y() > 270:
                pdf.add_page()
                
            # Extract lesson text
            lesson_parts = line.strip().split(':', 1)
            lesson_label = lesson_parts[0] + ":"
            lesson_content = lesson_parts[1].strip() if len(lesson_parts) > 1 else ""
            
            # Bold the entire lesson title (label + content)
            pdf.set_font("DejaVu", "B", 12)
            
            # Get the width of the lesson title
            title_width = pdf.get_string_width(lesson_label + " " + lesson_content)
            line_width = pdf.w - 2 * pdf.l_margin
            
            if title_width <= line_width:
                # If the title fits on one line, write it all in bold
                pdf.write(10, lesson_label + " " + lesson_content)
                pdf.ln(10)
            else:
                # If title is too long, handle wrapping
                words = (lesson_label + " " + lesson_content).split()
                x_position = pdf.l_margin
                pdf.set_x(x_position)
                
                for word in words:
                    word_width = pdf.get_string_width(word + " ")
                    if x_position + word_width > line_width:
                        pdf.ln()
                        x_position = pdf.l_margin
                        pdf.set_x(x_position)
                    
                    pdf.write(10, word + " ")
                    x_position += word_width
                
                pdf.ln(10)
            
        else:
            # Handle regular text
            if pdf.get_y() > 270:
                pdf.add_page()
                
            process_formatted_text(line, pdf)
            pdf.ln(10)

def process_formatted_text(text, pdf, starting_x=None):
    """Process text with Markdown formatting like bold (**text**) and italics (*text*)"""
    if starting_x is not None:
        pdf.set_x(starting_x)
    
    # Split the text into segments based on formatting
    segments = []
    current_pos = 0
    
    # Find all bold text (**text**)
    bold_pattern = re.compile(r'\*\*(.*?)\*\*')
    for match in bold_pattern.finditer(text):
        # Add text before the match
        if match.start() > current_pos:
            segments.append(('normal', text[current_pos:match.start()]))
        
        # Add the bold text
        segments.append(('bold', match.group(1)))
        current_pos = match.end()
    
    # Add any remaining text
    if current_pos < len(text):
        segments.append(('normal', text[current_pos:]))
    
    # If no formatting was found, add the whole text as normal
    if not segments:
        segments.append(('normal', text))
    
    # Process each segment with appropriate formatting
    line_width = pdf.w - 2 * pdf.l_margin
    x_position = pdf.get_x()
    
    for format_type, content in segments:
        # Set font based on format
        if format_type == 'bold':
            pdf.set_font("DejaVu", "B", 12)
        else:
            pdf.set_font("DejaVu", "", 12)
        
        # Process words with wrapping
        words = content.split()
        for word in words:
            word_width = pdf.get_string_width(word + " ")
            if x_position + word_width > line_width:
                pdf.ln()
                x_position = pdf.l_margin
                pdf.set_x(x_position)
            
            pdf.write(10, word + " ")
            x_position += word_width

def save_to_pdf(markdown_content, path, values):
    # Initialize PDF
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)  # Left, Top, Right margins
    pdf.add_page()

    # Add fonts
    try:
        pdf.add_font("DejaVu", "", "/app/documents/fonts/DejaVuSans.ttf", uni=True)  
        pdf.add_font("DejaVu", "B", "/app/documents/fonts/DejaVuSans-Bold.ttf", uni=True)
    except Exception as e:
        print(f"Error loading fonts: {e}")
        exit(1)

    # Add title
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, f'{values["course_name"]} Summary', ln=True, align="C")
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='L')
    pdf.ln(5)

    # Process Markdown
    try:
        process_markdown(markdown_content, pdf)
    except Exception as e:
        print(f"Error during markdown processing: {e}")
        exit(1)

    # Save PDF
    try:
        pdf.output(path)
        print("✅ PDF generated successfully!")
    except Exception as e:
        print(f"Error saving PDF: {e}")
        exit(1)