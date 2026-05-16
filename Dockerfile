# Use official Python runtime as a parent image
FROM python:3.10-slim

# Install git, which is required for the semantic analyzer
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
# Install CPU-only PyTorch first to save massive disk space (~2GB) and RAM
RUN pip install --no-cache-dir torch>=2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# IMPORTANT OPTIMIZATION:
# Pre-download the GraphCodeBERT model into the Docker image during the build process.
# This prevents the 500MB download from occurring on the first request, ensuring lightning-fast cold starts.
RUN python -c "from coderipple.modules.semantic_analyzer import GraphCodeBERTEmbedder; GraphCodeBERTEmbedder.get()"

# Ensure Flask knows it's behind a proxy
ENV PORT 8080

# CPU Optimizations to prevent thread thrashing on 2 vCPUs
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2

# Run gunicorn with 1 worker and 8 threads. 
# We set timeout to 0 because Cloud Run handles scaling and timeouts dynamically.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 coderipple.api.api:app
