import re
from typing import Literal
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, List, TypedDict
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, MessagesState, StateGraph

def create_embeddings():
    ''' Function to create vector embeddings '''
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    return ollama_embeddings

connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"
embeddings = create_embeddings()

vectorstore = PGVector(embeddings=embeddings, collection_name="lesson_content", connection=connection_string)

# Define prompt for question-answering
template = """You are an expert at summarizing programming education content. Your task is to create clear, concise summaries of programming lessons that capture the essential concepts.
Based on user request: summary {request}
Lesson Context:
{context}

For each lesson in the provided content:
1. Extract the lesson name
2. Identify the 3 most important concepts or techniques taught in the lesson
3. Create a summary with the lesson name followed by at most 3 bullet points

Format your summary as:
- Lesson: [LESSON NAME]
  • [First key concept with brief explanation]
  • [Second key concept with brief explanation]
  • [Third key concept with brief explanation]

Keep each bullet point concise and focused on one main idea. Include complexity analysis when relevant."""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOllama(model="llama3.2", temperature=0, base_url="http://localhost:11434")

# Define state for application
class State(MessagesState):
    question: str
    query: str
    context: List[Document]
    answer: str
    course_name: str
    course_id: str

# --- Utilites ---
def extract_course_info(input_string):
    # Regular expression to match the course name, ID, and regenerate flag
    pattern = r"course name: (.*?), id: (.*?)"
    
    # Search the string for matches
    match = re.search(pattern, input_string, re.IGNORECASE)
    
    if match:
        # Extracted groups
        course_name = match.group(1)
        course_id = match.group(2)
        return {
            "course_name": course_name,
            "course_id": course_id,
        }
    else:
        raise ValueError("Input string does not match the expected format.")

def extract_message(state: State):
    print("-------- EXTRACT MESSAGE ---------")
    message_content = state["messages"][-1].content
    extract_values = extract_course_info(message_content)
    print(extract_values)
    return {
        "course_name": extract_values["course_name"],
        "course_id": extract_values["course_id"],
    }
    
def retrieve(state: State):
    course_name = state["course_name"]
    retrieved_docs = vectorstore.as_retriever(search_kwargs={'k': 50, 'filter': {'course_name': course_name}})
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"request": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([extract_course_info, retrieve, generate])
graph_builder.add_edge(START, "extract_course_info")
graph = graph_builder.compile()