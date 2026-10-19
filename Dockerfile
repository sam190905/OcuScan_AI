# ==============================================================================
# OcuScan AI — Production Multi-Label Inference Container
# ==============================================================================

FROM python:3.11-slim

# Set environment optimization flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

# Install minimal OS dependencies for image processing & security certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements (using pre-built wheels for fast install)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py class_indices.json ./
COPY static/ ./static/
COPY templates/ ./templates/

# Ensure models directory exists
RUN mkdir -p models

# Copy local model weights if present (otherwise downloaded via startup hook)
COPY models/ ./models/

EXPOSE 8000

# Launch production ASGI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
