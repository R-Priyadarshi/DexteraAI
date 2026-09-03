# ============================================================
#  DexteraAI — Industrial Grade Container (Production)
# ============================================================

# ──── Stage 1: Web Builder ────
FROM node:20-slim AS web-builder
WORKDIR /app/web
COPY apps/web/package.json apps/web/package-lock.json* ./
RUN npm ci
COPY apps/web/ ./
RUN npm run build

# ──── Stage 2: Python Builder ────
FROM python:3.11-slim AS py-builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --no-cache-dir wheel && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -e ".[training]"

# ──── Stage 3: Production Runtime ────
FROM python:3.11-slim AS runtime

LABEL maintainer="DexteraAI Platform Team"
LABEL version="0.1.0-ultra"
LABEL description="Industrial-grade gesture intelligence platform"

WORKDIR /app

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from wheels
COPY --from=py-builder /app/wheels /app/wheels
RUN pip install --no-cache-dir /app/wheels/*

# Copy Source and Artifacts
COPY core/ core/
COPY training/ training/
COPY backend/ backend/
COPY dextera.py .
COPY __main__.py .
COPY pyproject.toml .
COPY scripts/healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh
COPY --from=web-builder /app/web/out /app/web/dist

# Secure Environment
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Production Security
RUN groupadd -r dextera && useradd -r -g dextera dextera
RUN chown -R dextera:dextera /app
USER dextera

# Health Check
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

ENTRYPOINT ["python", "dextera.py"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
