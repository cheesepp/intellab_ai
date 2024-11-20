FROM langchain/langgraph-api:3.11



ADD . /deps/IntellLab_AI

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"agent": "/deps/IntellLab_AI/agents/agent.py:graph"}'

WORKDIR /deps/IntellLab_AI