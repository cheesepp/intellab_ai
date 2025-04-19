from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = OpenAI(model="dall-e-3")
prompt = ChatPromptTemplate.from_template(
    template="Generate a detailed prompt to generate an image based on the following description: {image_desc}",
)
chain = prompt | llm

chain.invoke({"image_desc": "a cat on a stone"})