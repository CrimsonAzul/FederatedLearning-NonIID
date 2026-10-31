# One image for every role: prepare (data), server and client.
# The base image ships PyTorch with CUDA 12.4 and also runs on machines without a GPU.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir "numpy<2.3" "flask>=3.0" "requests>=2.31" "pyyaml>=6.0"

COPY fl ./fl
COPY server ./server
COPY client ./client
COPY scripts ./scripts
COPY configs ./configs
