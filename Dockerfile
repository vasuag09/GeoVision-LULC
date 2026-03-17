FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and rasterio
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

WORKDIR /workspace
COPY . /workspace

# Sync dependencies using uv
RUN uv sync --frozen

# Expose Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["/workspace/.venv/bin/streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
