# Dockerfile

# 1. Base Image: Choose a CUDA version compatible with your host driver and onnxruntime-gpu
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1 \
    # Poetry configuration
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    # Set locale
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 3. Install Python and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# 4. Set up work directory
WORKDIR /app

# 5. Copy dependency files and install dependencies
# Copy only necessary files first to leverage Docker cache
COPY pyproject.toml poetry.lock ./
# Install dependencies including onnxruntime-gpu
# --no-root is important if pdf2zh is listed as develop = true
RUN poetry install --no-dev --no-root

# 6. Copy the rest of the application code
COPY . /app

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
