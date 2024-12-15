from langchain_core.documents import Document
from typing import TypedDict
from agents.utils.nodes import document_search, finalize_response, generate, grade_generation_v_documents_and_question, transform_query, web_search
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, add_messages

from agents.utils.state import GraphState
    
class GraphConfig(TypedDict):
    max_retries: int
    
# Define graph

workflow = StateGraph(GraphState, config_schema=GraphConfig)

# Define the nodes
workflow.add_node("document_search", document_search)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)
workflow.add_node("finalize_response", finalize_response)

# Build graph
workflow.set_entry_point("document_search")
workflow.add_edge("document_search", "generate")
workflow.add_edge("transform_query", "document_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("finalize_response", END)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question
)

# Compile
graph = workflow.compile()

# VERBOSE = False
# inputs = {"messages": [("human", "how do i calculate sum by groups")]}
# for output in graph.stream(inputs):
#     print(output)
#     print("\n---\n")