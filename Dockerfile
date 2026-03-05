# ── Stage 1: Build dependencies ─────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Production image ──────────────────────────────
FROM python:3.11-slim

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
COPY train.py .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose the serving port
EXPOSE 8000

# ECS health check (ALB also checks /health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production server: multiple workers, graceful shutdown
# Workers = 2 * vCPU + 1 (tune via ECS task CPU allocation)
CMD ["uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-graceful-shutdown", "30", \
     "--log-level", "info"]
