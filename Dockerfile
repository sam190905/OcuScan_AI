# ==============================================================================
# OcuScan AI — Production Multi-Label Inference Container (Hugging Face Spaces)
# ==============================================================================

FROM python:3.11-slim

# Set environment optimization flags & default Hugging Face Spaces port (7860)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

# Install minimal OS dependencies for image processing & security certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up standard non-root user (UID 1000) for Hugging Face Spaces security
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python requirements
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files and assets
COPY --chown=user:user . $HOME/app

# Ensure models directory exists with correct permissions
RUN mkdir -p $HOME/app/models

EXPOSE 7860

# Launch production ASGI server on Hugging Face port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
