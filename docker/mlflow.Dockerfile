FROM ghcr.io/mlflow/mlflow:v3.18.0
RUN pip install --no-cache-dir psycopg2-binary
