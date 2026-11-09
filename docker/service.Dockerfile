# Inference service: CPU-only, no torch. ONNX Runtime + tokenizers + FastAPI (~300 MB image).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 TICKETFLOW_ROOT=/app
WORKDIR /app

COPY pyproject.toml params.yaml ./
COPY src ./src
RUN pip install .

RUN useradd --create-home app && mkdir -p /data/state /tmp/ticketflow-models && chown -R app /data /tmp/ticketflow-models
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "ticketflow.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
