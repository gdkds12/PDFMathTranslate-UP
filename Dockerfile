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

# 6. Install the project and its dependencies (this might install CPU onnxruntime)
RUN pip install -v --no-cache-dir .

# 7. Explicitly uninstall any CPU onnxruntime and reinstall/ensure onnxruntime-gpu
#    This attempts to fix the potential conflict or overwrite issue.
RUN pip uninstall -y onnxruntime onnxruntime-gpu # Uninstall both to be sure
RUN pip install -v --no-cache-dir onnxruntime-gpu==1.20.0 # Reinstall the desired GPU version

# 8. Install other direct dependencies of pdf2zh from pyproject.toml, excluding onnxruntime and onnxruntime-gpu
#    babeldoc and other dependencies will pull their own sub-dependencies (hopefully not conflicting onnxruntime)
RUN pip install -v --no-cache-dir \
    requests \
    "pymupdf<1.25.3" \
    tqdm \
    tenacity \
    numpy \
    ollama \
    xinference-client \
    deepl \
    "openai>=1.0.0" \
    "azure-ai-translation-text<=1.0.1" \
    gradio \
    huggingface_hub \
    tencentcloud-sdk-python-tmt \
    "pdfminer.six>=20240706" \
    "gradio_pdf>=0.0.21" \
    pikepdf \
    "peewee>=3.17.8" \
    fontTools \
    "babeldoc>=0.1.22,<0.3.0" \
    rich \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    aiofiles \
    python-dotenv

# 9. Expose the port the app runs on
EXPOSE 8000

# 10. Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
