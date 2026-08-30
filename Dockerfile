# =============================================================================
# PhosphogypsumBot: Multi-Stage Production Docker Image
# =============================================================================

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml /app/
COPY pgloop/__init__.py /app/pgloop/__init__.py

# Install PyTorch CPU and core package dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -e ".[viz,ai,docling,stochastic_dynamics]"

# Copy application source code
COPY . /app

# Ensure package is installed in editable mode
RUN pip install -e .

# Expose Streamlit dashboard port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: Launch interactive Streamlit dashboard
CMD ["streamlit", "run", "pgloop/visualization/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
