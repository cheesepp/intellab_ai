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

# Copy the pre-built wheel file to the container
COPY build/*.whl /app/

# Install the wheel package
RUN pip install /app/*.whl --no-cache-dir

# Expose FastAPI port
EXPOSE 8106

# Run the FastAPI application
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8006"]

