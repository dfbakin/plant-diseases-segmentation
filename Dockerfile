# Build on top of Vast.ai's base image (includes SSH, Jupyter, Supervisor, etc.)
# See: https://hub.docker.com/r/vastai/base-image
FROM vastai/base-image:cuda-12.8.1-auto

# Install additional system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with system CUDA (no bundled nvidia-* packages)
RUN . /venv/main/bin/activate && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies (--no-deps to avoid pulling nvidia-* transitively)
COPY requirements-docker.txt /tmp/requirements.txt
RUN . /venv/main/bin/activate && \
    uv pip install --no-deps -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Install DVC for data versioning
RUN . /venv/main/bin/activate && \
    uv pip install dvc[s3]

# Set PYTHONPATH so 'src' module is importable without pip install -e
ENV PYTHONPATH="/workspace/plant-diseases-segmentation:${PYTHONPATH}"

WORKDIR /workspace
