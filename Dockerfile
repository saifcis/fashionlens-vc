
FROM ghcr.io/astral-sh/uv:0.13.10 AS uv

FROM python:3.12.3-slim-bookworm

WORKDIR /app

# Python and ML configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/huggingface \
    HF_HUB_CACHE=/opt/huggingface/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PORT=8080

# Install uv
COPY --from=uv /uv /uvx /usr/local/bin/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install locked Python dependencies
COPY pyproject.toml uv.lock .python-version ./

RUN uv sync \
    --locked \
    --no-dev \
    --no-install-project \
    --no-cache

ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY app/ ./app/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
COPY streamlit_app.py ./
COPY .streamlit/ ./.streamlit/

# Copy catalog, images and embeddings
COPY data/ ./data/

# Copy pretrained CLIP model
COPY deployment/hf-cache/ /opt/huggingface/

# Create non-root application user
RUN useradd --create-home --uid 10001 fashionlens

USER fashionlens

EXPOSE 8080

# Start application
CMD ["sh", "-c", "exec streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless=true"]
