# DexteraAI — Training & Inference Container
# Multi-stage build for minimal production image

# ──── Stage 1: Builder ────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[training]"

# ──── Stage 2: Runtime ────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source
COPY core/ core/
COPY training/ training/
COPY backend/ backend/
COPY dextera.py .
COPY __main__.py .
COPY pyproject.toml .

# Create non-root user
RUN useradd -m -u 1000 dextera
USER dextera

# Default: show help
ENTRYPOINT ["python", "dextera.py"]
CMD ["--help"]
