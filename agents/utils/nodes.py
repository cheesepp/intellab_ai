from langchain_core.messages import BaseMessage, AIMessage, convert_to_messages
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import Literal
from langchain_ollama import ChatOllama, OllamaEmbeddings
from agents.utils.models import GradeAnswer, GradeHallucinations
from agents.utils.prompts import ANSWER_GRADER_PROMPT, HALLUCINATION_GRADER_PROMPT, QUERY_REWRITER_PROMPT, RAG_PROMPT
from agents.utils.state import GraphState
from agents.utils.vectordb import get_retriever
from langchain_community.tools import TavilySearchResults

MAX_RETRIES=3

# LLM / Retriever / Tools
llm = ChatOllama(model="llama3.2", base_url="http://host.docker.internal:11434")
# llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

retriever = get_retriever()
tavily_search_tool = TavilySearchResults(max_results=3)


def document_search(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = convert_to_messages(state["messages"])[-1].content

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "web_fallback": True}

def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retries = state["retries"] if state.get("retries") is not None else -1

    rag_chain = RAG_PROMPT | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"retries": retries + 1, "candidate_answer": generation}

def transform_query(state: GraphState):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]

    # Re-write question
    query_rewriter = QUERY_REWRITER_PROMPT | llm | StrOutputParser()
    better_question = query_rewriter.invoke({"question": question})
    return {"question": better_question}

def web_search(state: GraphState):
    print("---RUNNING WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    search_results = tavily_search_tool.invoke(question)
    search_content = "\n".join([d["content"] for d in search_results])
    documents.append(Document(page_content=search_content, metadata={"source": "websearch"}))
    return {"documents": documents, "web_fallback": False}

def grade_generation_v_documents_and_question(state: GraphState, config) -> Literal["generate", "transform_query", "web_search", "finalize_response"]:
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["candidate_answer"]
    web_fallback = state["web_fallback"]
    retries = state["retries"] if state.get("retries") is not None else -1
    max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

    # this means we've already gone through web fallback and can return to the user
    if not web_fallback:
        return "finalize_response"

    print("---CHECK HALLUCINATIONS---")
    hallucination_grader = HALLUCINATION_GRADER_PROMPT | llm.with_structured_output(GradeHallucinations)
    hallucination_grade: GradeHallucinations = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    # Check hallucination
    if hallucination_grade.binary_score == "no":
        return "generate" if retries < max_retries else "web_search"

    print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

    # Check question-answering
    print("---GRADE GENERATION vs QUESTION---")

    answer_grader = ANSWER_GRADER_PROMPT | llm.with_structured_output(GradeAnswer)
    answer_grade: GradeAnswer = answer_grader.invoke({"question": question, "generation": generation})
    if answer_grade.binary_score == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "finalize_response"
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "transform_query" if retries < max_retries else "web_search"


def finalize_response(state: GraphState):
    print("---FINALIZING THE RESPONSE---")
    return {"messages": [AIMessage(content=state["candidate_answer"])]}