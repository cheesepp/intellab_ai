from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableConfig,
    RunnablePassthrough,
)
from core import get_model, settings
from langchain_core.output_parsers import StrOutputParser

def call_model(state: MessagesState, config: RunnableConfig):
    messages = state["messages"]
    template = """You are an assistant about making a title (with no subject) from a human's sentence. Please make a title concisely no greater than 10 words
    
    Human: {human_message}
    Just return the title, do not say anything."""
    summary_title_prompt = ChatPromptTemplate.from_template(template)
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    title_assistant = summary_title_prompt | model | StrOutputParser()
    summary_response = title_assistant.invoke({"human_message": messages[-1].content})
    return {"messages": [summary_response]}


# Define the graph
agent = StateGraph(MessagesState)
agent.add_node("model", call_model)
agent.set_entry_point("model")

# Always END after blocking unsafe content
agent.add_edge("model", END)

title_generator = agent.compile()
