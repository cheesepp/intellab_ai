import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from schema.models import GroqModelName

load_dotenv()

from agents.agents import DEFAULT_AGENT, agents  # noqa: E402

agent = agents["lesson_chatbot"].graph


async def main() -> None:
    chunks = []
    inputs = {"messages": [("human", "Lesson: Introduction to Circular Linked List Lesson_id: 4cd8171b-fbfb-4ac2-bd58-4d59d056750d Question: Explain string theory")]}
    async for chunk in agent.astream(inputs):
        chunks.append(chunk)
        print(chunk, end="|", flush=True)
    # result = await agent.ainvoke(
    #     inputs,
    #     config=RunnableConfig(configurable={"thread_id": uuid4(), "model": GroqModelName.LLAMA_33_70B}),
    # )
    # result["messages"][-1].pretty_print()

    # Draw the agent graph as png
    # requires:
    # brew install graphviz
    # export CFLAGS="-I $(brew --prefix graphviz)/include"
    # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    # pip install pygraphviz
    #
    # agent.get_graph().draw_png("agent_diagram.png")


if __name__ == "__main__":
    asyncio.run(main())
