from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.retrieval import create_retrieval_chain
from langchain_nomic import NomicEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
import os
from langchain_core.runnables import (
    RunnableLambda,
    RunnableConfig,
)
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Literal, Annotated, TypedDict
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import (
    AnyMessage,
)
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, SystemMessage
from sqlalchemy import create_engine, text

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment, parse_llama_guard_output, parse_llama_guard_relevant_topic
from core import settings
from core.llm import get_model
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableParallel
from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from langchain_core.stores import InMemoryByteStore
from langchain.storage import LocalFileStore
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_core.vectorstores import VectorStore

from service.utils import normalize_uid

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# DB_CONNECTION_STRING = "postgresql://postgres:123456@localhost:5433/intellab-db"
# OLLAMA_HOST="http://localhost:11434"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# We will add a `summary` attribute (in addition to `messages` key,
# which MessagesState already has)
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    safety: LlamaGuardOutput
    is_relevant: LlamaGuardOutput
    courses_retriever: List[Document]
    problems_retriever: List[Document]
    user_problems_retriever: List[Document]
    user_courses_retriever: List[Document]
# We will use this model for both the conversation and the summarization
# model = ChatOllama(model="llama3.2")

# template = """You are an assistant for question-answering tasks that focus on algorithms and data structures. Use the following pieces of retrieved context to answer the question. Each question, you must explain deeply and understandable. If you don't know the answer, just say that you don't know. 
# Answer questions based on conversation history:
# Summary: {summary}
# Current conversation: {conversation}

# When you recommend some courses, always give the url along which point to that course with endpoint is course_id and appended with "https://localhost:3000/courses/" (just when recommending).
# Please follow strictly with the provided context, do not recommend any courses outside.
# Question: {question} 
# Context: {context} 
# Answer: Just response the question, do not say 'Based on' or something similar."""
template = """
Hey there! I'm your friendly programming buddy who absolutely loves algorithms and data structures. Think of me as that enthusiastic friend who gets excited about coding and wants to help you succeed!

Here's what we've been talking about:
Summary: {summary}
Recent chat: {conversation}

Your question: {question}

Resources I can share:
Courses: {courses}
Problems: {problems}
- **Your solved problems or problems you interacted with:** {user_problems}
- **Your courses or learning progress:** {user_courses}

My approach:
- I'll chat with you like we're grabbing coffee and talking code
- Break down tricky concepts into bite-sized pieces you can actually use
- Share real examples that make sense in the real world  
- Mention time/space complexity when it actually helps (not just to sound smart!)
- **When recommending, only suggest problems that you have NOT solved yet and courses that you have NOT completed.**
- **When recommending problems, prioritize the newest ones based on their 'Created At' date.**
- **When recommending courses, prioritize those with the highest 'Rating' and the most recent 'Created At' date.**
- Recommend problems like this: [Problem Title](https://www.intellab.site/problems/problem_id)
- Suggest courses like this: [Course Title](https://www.intellab.site/course/course_id)
- I'll only suggest stuff from what I actually have access to
- If I can't help with something, I'll be honest but keep you motivated

Let's keep this fun and natural - no robot talk here!"""

# loader = CSVLoader(file_path='./documents/courses.csv')
# problem_loader = CSVLoader(file_path='./documents/problems.csv')
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")

embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=NOMIC_API_KEY)
# Create an index using the loaded documents
index_creator = VectorstoreIndexCreator(embedding=embeddings)
# docsearch = index_creator.from_loaders([problem_loader, loader])
prompt = ChatPromptTemplate.from_template(template)
# course_vectorstore = PGVector(embeddings=embeddings, collection_name="courses", connection=DB_CONNECTION_STRING, use_jsonb=True, )
# problem_vectorstore = PGVector(embeddings=embeddings, collection_name="problems", connection=engine, use_jsonb=True)

engine = create_engine(DB_CONNECTION_STRING)

CACHE_DIR = "./.cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_solved_problems_by_user(user_id: str) -> List[Document]:
    """
    Lấy danh sách Document của các problem mà user đã solve
    """
    # query = """
    #     SELECT DISTINCT problem_id
    #     FROM problem_submissions
    #     WHERE user_id = :user_id AND is_solved = TRUE
    # """
    query = """SELECT p.problem_id, p.problem_name 
    FROM public.problems p 
    JOIN problem_submissions ps
    on p.problem_id = ps.problem_id
    where ps.user_id = :user_id AND is_solved = TRUE"""

    with engine.connect() as conn:
        results = conn.execute(text(query), {"user_id": user_id})
        documents = []
        for row in results:
            record = row._mapping
            problem_id = record["problem_id"]
            problem_name = record["problem_name"]
            content = f"Problem ID: {problem_id} - Name: {problem_name}"
            doc = Document(
                page_content=content,  # Convert UUID to string
                metadata={
                    "problem_id": str(record["problem_id"]),  # Also add as metadata
                    "problem_name": str(record["problem_name"]),
                }
            )
            documents.append(doc)
        return documents

def get_user_courses(user_id: str) -> List[Document]:
    """
    Truy vấn các khoá học của user, trả về List[Document]
    """
    query = """
        SELECT
            uc.course_id,
            uc.status,
            uc.progress_percent,
            c.course_name
        FROM
            user_courses uc
        JOIN
            courses c ON uc.course_id = c.course_id
        WHERE
            uc.user_uid = :user_id
    """
    with engine.connect() as conn:
        results = conn.execute(text(query), {"user_id": user_id})
        documents = []
        for row in results:
            r = row._mapping
            course_id = r["course_id"]
            course_name = r["course_name"]
            status = r["status"]
            progress = float(r["progress_percent"])
            content = f"Course {course_id} Name: {course_name} - Status: {status} - Progress: {progress:.2f}%"
            doc = Document(
                page_content=content,
                metadata={
                    "course_id": str(r["course_id"]),
                    "status": r["status"],
                    "progress_percent": float(r["progress_percent"]),
                }
            )
            documents.append(doc)
        return documents
def get_all_problems() -> List[Document]:
    """
    Lấy tất cả các problem đã được publish, trả về List[Document]
    """
    query = """
        SELECT problem_id, problem_name, problem_level, score, created_at FROM public.problems
        Where is_published = 'TRUE'
        ORDER BY created_at DESC 
    """
    with engine.connect() as conn:
        results = conn.execute(text(query))
        documents = []
        for row in results:
            r = row._mapping
            content = f"Problem ID: {r['problem_id']} - Name: {r['problem_name']} - Level: {r['problem_level']} - Score: {r['score']} - Created At: {r['created_at']}"
            print(f"======== PROBLEM DOCS {content}")
            doc = Document(
                page_content=content,
                metadata={
                    "problem_id": str(r["problem_id"]),
                    "problem_name": r["problem_name"],
                    "level": r["problem_level"],
                }
            )
            documents.append(doc)
        return documents

def get_all_courses() -> List[Document]:
    """
    Lấy tất cả các khoá học, trả về List[Document]
    """
    query = """
        SELECT course_id, course_name, description, level, price, unit_price, average_rating, review_count, score, created_at FROM public.courses
        Where is_available = true
        ORDER BY created_at, average_rating DESC 
    """
    with engine.connect() as conn:
        results = conn.execute(text(query))
        documents = []
        for row in results:
            r = row._mapping
            description = r.get("description", "")  # Lấy description, nếu không có thì trả về chuỗi rỗng
            content = f"Course: {r['course_name']} - Description: {description} - Level: {r['level']} - Price: {r['price']} {r.get('unit_price', '')} - Rating: {r['average_rating']} ({r['review_count']} reviews) - Score: {r['score']} - Created At: {r['created_at']}"
            print(f"======== COURSE DOCS {content}")
            doc = Document(
                page_content=content,
                metadata={
                    "course_id": str(r["course_id"]),
                    "course_name": r["course_name"],
                }
            )
            documents.append(doc)
        return documents
       
def build_chroma_vectorstore(docs: List[Document], collection_name: str):
    # Cache (giảm chi phí embed)
    base_embeddings = embeddings

    persist_directory = os.path.join(CACHE_DIR, "chroma_db", collection_name)
    cache_store_path = os.path.join(CACHE_DIR, "embeddings", collection_name)

    # cache_store = InMemoryByteStore()
    fs_store = LocalFileStore(root_path=cache_store_path)
    # Sử dụng namespace để tạo cache riêng biệt
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        base_embeddings, fs_store, namespace=collection_name
    )

    # Vector store trong RAM
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=cached_embeddings,
        persist_directory=persist_directory
    )

    # vectorstore.add_documents(docs)
    vectorstore.persist() # Lưu các thay đổi xuống đĩa

    return vectorstore

# 1. Khai báo biến toàn cục, khởi tạo là None
GLOBAL_PROBLEM_VECTORSTORE = None
GLOBAL_COURSE_VECTORSTORE = None
_vector_stores_initialized = False

def initialize_global_vector_stores():
    """
    Hàm này chỉ thực hiện khởi tạo vector store ở lần gọi đầu tiên.
    Các lần gọi sau sẽ không làm gì cả.
    """
    global GLOBAL_PROBLEM_VECTORSTORE, GLOBAL_COURSE_VECTORSTORE, _vector_stores_initialized
    
    # Dùng cờ để kiểm tra, nếu đã khởi tạo thì bỏ qua
    if _vector_stores_initialized:
        return

    print("Initializing vector stores for the first time...")

    # Lấy tất cả dữ liệu từ DB
    all_problems_docs = get_all_problems()
    all_courses_docs = get_all_courses()

    # Xây dựng vector store và gán vào biến toàn cục
    GLOBAL_PROBLEM_VECTORSTORE = build_chroma_vectorstore(all_problems_docs, collection_name="all_problems")
    GLOBAL_COURSE_VECTORSTORE = build_chroma_vectorstore(all_courses_docs, collection_name="all_courses")
    
    # Đặt cờ thành True để các lần gọi sau không chạy lại
    _vector_stores_initialized = True
    print("Vector stores initialized successfully.")

# --- END: KHỞI TẠO VECTOR STORE ---

def retrieve_context(vectorstore: VectorStore, query: str, k: int = 5) -> List[Document]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

async def retrieve_vectordb(state: State, config: RunnableConfig):
    course_vectorstore = PGVector(embeddings=embeddings, collection_name="courses", connection=DB_CONNECTION_STRING, use_jsonb=True, )
    problem_vectorstore = PGVector(embeddings=embeddings, collection_name="problems", connection=DB_CONNECTION_STRING, use_jsonb=True)
    
     # Đảm bảo vector store toàn cục đã được khởi tạo
    initialize_global_vector_stores()
    
    message = state["messages"][-1].content
    userid = config["configurable"].get("user_id")
    user_id = normalize_uid(userid)
    print(f"NORMALIZE UID {userid} -> {user_id}")
    # solved_problem_ids = get_solved_problems_by_user(user_id=config["configurable"].get("user_id"))
    # if not solved_problem_ids:
    #     # If no solved problems, return empty list
    #     user_problem_docs = []
    # else:
    #     user_problem_vectorstore = build_chroma_vectorstore(solved_problem_ids, "user_problem_embed")
    #     user_problem_docs = retrieve_context(user_problem_vectorstore, message)
    
    # course_ids = get_user_courses(user_id=config["configurable"].get("user_id"))
    # if not course_ids:
    #     # If no courses, return empty list
    #     user_course_docs = []
    # else:   
    #     user_course_vectorstore = build_chroma_vectorstore(course_ids, "user_course_embed")
    #     user_course_docs = retrieve_context(user_course_vectorstore, message)

    # 2. Lấy trực tiếp tài liệu của user mà không cần tạo vector store
    # Dữ liệu này đủ nhỏ để đưa thẳng vào prompt
    user_problem_docs = get_solved_problems_by_user(user_id=user_id)
    user_course_docs = get_user_courses(user_id=user_id)

    # Truy vấn context từ vector store chung
    problem_docs = retrieve_context(GLOBAL_PROBLEM_VECTORSTORE, message)
    course_docs = retrieve_context(GLOBAL_COURSE_VECTORSTORE, message)
   
    # course_retriever = course_vectorstore.as_retriever()
    # problem_retriever = problem_vectorstore.as_retriever()
    # course_docs = course_retriever.invoke(message)
    # problem_docs = problem_retriever.invoke(message)
    return {"courses_retriever": course_docs, "problems_retriever": problem_docs, "user_problems_retriever": user_problem_docs, "user_courses_retriever": user_course_docs}

# Define the logic to call the model
async def acall_model(state: State, config: RunnableConfig):
    print(f'------- CALL MODEL {config["configurable"].get("model", settings.DEFAULT_MODEL)}----------')
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    # If a summary exists, we add this in as a system message
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    # print(f"--------- {messages}-----------")
    m_as_string = "\n\n".join([message.content for message in messages])
    courses_context = state["courses_retriever"]
    problems_context = state["problems_retriever"]
    user_courses_context = state["user_courses_retriever"]
    user_problems_context = state["user_problems_retriever"]
    qa_chain = (
        # RunnableParallel(courses=courses_context, problems=problems_context) |
        {
            "courses": RunnableLambda(lambda _: courses_context),
            "problems": RunnableLambda(lambda _: problems_context),
            "user_courses": RunnableLambda(lambda _: user_courses_context),
            "user_problems": RunnableLambda(lambda _: user_problems_context),
            "summary": RunnableLambda(lambda _: summary),
            "conversation": RunnableLambda(lambda _: m_as_string),
            "question": RunnableLambda(lambda _: messages[-1].content)
        }
        | prompt
        | model
        # | StrOutputParser()
    )
    
    print(f"------- doc search {courses_context} ----------")
    response = await qa_chain.ainvoke(messages[-1].content, config)
    # response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# We now define the logic for determining whether to end or summarize the conversation
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    # Otherwise we can just end
    return END


async def summarize_conversation(state: State, config: RunnableConfig):
    print(f"------- SUMMARIZE CONVERSATION ----------")
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    # First, we summarize the conversation
    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    summary_messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(summary_messages)
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)

def format_is_relevant_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"{safety.unsafe_response}"
    )
    return AIMessage(content=content)

async def llama_guard_input(state: State, config: RunnableConfig) -> State:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}

async def llama_guard_output(state: State, config: RunnableConfig) -> State:
    checking_template = """You are an expert at determining whether an input is relevant to programming or casual conversation that should be allowed in a programming-focused chatbot.

Analyze the 'Latest User Input' based on the 'Conversation Summary' and 'Recent Conversation History' to understand the context.

Conversation Summary:
{summary}

Recent Conversation History:
{conversation_history}

Latest User Input: {question}

Your task is to:
1. Analyze the latest user input in the context of the conversation history.
2. Determine if it's relevant to programming topics OR is appropriate casual conversation.
3. Provide a clear response.

For each input, respond with:
- FIRST LINE: Either "relevant" or "irrelevant" (lowercase, no additional words)
- SECOND LINE: A brief explanation of your decision. If irrelevant, respond friendly and advise the user back to programming topics, starting with "Sorry, but...".

Classify as RELEVANT:
- Programming languages, algorithms, data structures (e.g., "queue", "stack").
- Software development, computer science concepts, coding problems.
- System design, development tools, questions about programming courses.
- **Any question mentioning "problem", "problems", "course", "courses", or academic terms related to mathematics, computer science, algorithms, or programming.**
- **Questions about the user's personal learning status, course progress, or solved problems on this platform (e.g., "my courses learning status", "what problems have I solved?").**
- Greetings and casual conversation starters ("Hi", "How are you?").
- Follow-up questions that are part of a programming conversation (e.g., if we were talking about data structures, "queue" is relevant).
- Career advice in programming.
- Any ambiguous question that could reasonably relate to programming, especially given the conversation history.

Examples:
Input: "How do I implement a binary search in Python?"
Output:
relevant
This question directly relates to implementing an algorithm in a programming language.

Input: "my courses learning status"
Output:
relevant
This question is about the user's progress on the platform's programming courses.

Input: "problems"
Output:
relevant
The user is likely referring to programming problems or exercises.

Input: "courses"
Output:
relevant
The user is likely referring to programming or computer science courses.

Input: "What's the best recipe for chocolate chip cookies?"
Output:
irrelevant
Sorry, but this question is about cooking/baking and has no connection to programming.
"""
    prompt = ChatPromptTemplate.from_template(checking_template)
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    checking_output = prompt | model
    messages = state["messages"]
    summary = state.get("summary", "No summary yet.")
    # Use the latest user message as the question
    question = messages[-1].content if messages else ""
    conversation_history = "\n\n".join(
        [
            f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
            for m in messages
        ]
    )

    response = await checking_output.ainvoke({
        "summary": summary,
        "conversation_history": conversation_history,
        "question": question
    })
    llama_guard = parse_llama_guard_relevant_topic(response.content)
    return {"is_relevant": llama_guard}

# async def llama_guard_output(state: State, config: RunnableConfig) -> State:
#     # Simple keyword-based check for obviously irrelevant content
#     message_content = state["messages"][-1].content
    
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
    
async def block_unsafe_content(state: State, config: RunnableConfig) -> State:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}

async def block_irrelevant_content(state: State, config: RunnableConfig) -> State:
    safety: LlamaGuardOutput = state["is_relevant"]
    return {"messages": [format_is_relevant_message(safety)]}


# Check for unsafe input and block further processing if found
def check_is_relevant(state: State) -> Literal["relevant", "irrelevant"]:
    is_relevant: LlamaGuardOutput = state["is_relevant"]
    match is_relevant.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "irrelevant"
        case _:
            return "relevant"

# Check for unsafe input and block further processing if found
def check_safety(state: State) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


# Define a new graph
workflow = StateGraph(State)

# Define the conversation node and the summarize node
workflow.add_node("retrieval", retrieve_vectordb)
workflow.add_node("conversation", acall_model)
# workflow.add_node("normal_conversation", normal_conversation)
workflow.add_node(summarize_conversation)
workflow.add_node("guard_input", llama_guard_input)
workflow.add_node("guard_output", llama_guard_output)
workflow.add_node("block_unsafe_content", block_unsafe_content)
workflow.add_node("block_unsafe_output_content", block_irrelevant_content)

# Set the entrypoint as conversation
workflow.set_entry_point("guard_input")

# workflow.add_edge("guard_input", "conversation")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "conversation",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

workflow.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "guard_output"}
)

workflow.add_conditional_edges(
    "guard_output", check_is_relevant, {"irrelevant": "block_unsafe_output_content", "relevant": "retrieval" }
)

workflow.add_edge("retrieval", "conversation")
workflow.add_edge("summarize_conversation", END)
# Always END after blocking unsafe content
workflow.add_edge("block_unsafe_content", END)
workflow.add_edge("block_unsafe_output_content", END)

global_chatbot = workflow.compile()