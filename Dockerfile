# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Install dependencies in an isolated layer so the final image only
# copies the compiled wheels, not the build toolchain.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed by some wheels (e.g. chromadb, docling)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy project source
COPY backend/   ./backend/
COPY frontend/  ./frontend/
COPY scripts/   ./scripts/

# chroma_db sits at the project root (/app/chroma_db), not inside backend/.
# retrieval.py resolves the path as Path(__file__).parent.parent / "chroma_db"
# which from /app/backend/retrieval.py → /app/chroma_db
RUN mkdir -p /app/chroma_db /app/data/raw_labels \
    && chown -R appuser:appuser /app

USER appuser

# HF_TOKEN must be supplied at runtime via --env or docker-compose env_file.
# Never bake secrets into the image.
ENV HF_TOKEN=""

# Uvicorn listens on 0.0.0.0 so Docker can forward the port.
# The app itself is always reached at http://localhost:8000 on the host.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run uvicorn from inside backend/ so that sibling imports
# (from graph import ..., from retrieval import ...) resolve correctly
# without requiring a package __init__.py.
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
