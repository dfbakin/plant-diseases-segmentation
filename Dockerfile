FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pipx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pipx install dvc[s3]

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

# Expose ports for Jupyter/MLflow if needed
EXPOSE 8888 5000

# Keep container alive by default
CMD ["tail", "-f", "/dev/null"]
