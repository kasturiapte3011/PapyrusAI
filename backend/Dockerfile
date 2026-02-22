FROM python:3.11-slim

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock* ./

# Install dependencies
RUN uv sync --no-dev

# Copy app source code
COPY . .

EXPOSE 8000

# Run FastAPI with controlled concurrency
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
