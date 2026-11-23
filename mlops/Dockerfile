FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create non-root user for container security
RUN useradd --create-home --shell /bin/bash appuser

COPY pyproject.toml README.md requirements-serve.txt ./
COPY shared ./shared
COPY modules ./modules
COPY app ./app

# Install CPU-only PyTorch to keep the serving image lightweight, then API dependencies
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install -r requirements-serve.txt && \
    pip install --no-deps --no-build-isolation -e .

RUN mkdir -p /app/storage && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
