# Dockerfile (Revised for PEP 621 / Hatchling, No Poetry Install)

# 1. Base Image: Use a -devel image for full CUDA toolkit and cuDNN support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Make sure pip's bin is in PATH if needed, though usually handled
    PATH=/root/.local/bin:$PATH \
    # Prevent pip from caching, reduces image size slightly
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# 3. Install Python, pip, and necessary build tools/libs
# Add build-essential if any dependencies require compilation
# Add git if any dependencies are fetched from git repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git build-essential libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# 4. Set up work directory
WORKDIR /app

# 5. Copy ALL project files first
# This includes pyproject.toml, README.md, the pdf2zh directory, etc.
COPY . /app

# 6. Install the project and its dependencies using pip
# Now pip/hatchling can find the 'pdf2zh' directory and other necessary files
RUN pip install --no-cache-dir .

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
