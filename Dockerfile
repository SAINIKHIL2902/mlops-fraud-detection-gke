# Multi-stage production Dockerfile for Real-Time Fraud Detection API
FROM python:3.11-slim as builder

WORKDIR /install

# Install build tools if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final lean runtime image
FROM python:3.11-slim

# Security: prevent running as root
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/bash -m appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ENVIRONMENT=production \
    PORT=8000

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /install /usr/local

# Copy application source code and pre-trained model artifacts
COPY app/ /app/app/
COPY models/ /app/models/
COPY pipeline/ /app/pipeline/

# Ensure non-root ownership
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Container healthcheck using standard library urllib (no curl dependency required)
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live')" || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "30"]

