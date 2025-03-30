FROM python:3.12.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app
USER root
ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

# Install PostgreSQL client libraries
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the .env file into the container
COPY docker.env /app/.env

# Copy the source code into the container
COPY src /app/src
COPY src/researchs/ ./researchs/
COPY src/documents/ ./documents/

COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# Install dependencies
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-install-project --no-dev

# Expose the port 
EXPOSE 8106

# Set the default command to run your application
CMD ["python", "/app/src/run_service.py"]
