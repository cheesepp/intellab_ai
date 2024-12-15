FROM langchain/langgraph-api:3.11

# Set the working directory
WORKDIR /deps/IntellLab_AI

# Update packages and install dependencies
RUN apt-get update && apt-get install -y build-essential python3-dev libpq-dev gcc python3-distutils make

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Install dependencies without the package itself
RUN poetry install --no-root --only main

# Copy the rest of the application code
COPY . .

# Install the package
RUN poetry install --only main
RUN poetry show langchain-community
# Set environment variables
ENV LANGSERVE_GRAPHS='{"agent": "/deps/IntellLab_AI/agents/agent.py:graph"}'

# Specify the command to run
CMD ["python", "your_main_script.py"]