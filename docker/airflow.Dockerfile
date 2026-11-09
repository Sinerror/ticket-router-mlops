# Airflow with the ticketflow package. Retraining runs on the host GPU (compose: gpus: all);
# torch is the same CUDA build as in the host venv (uv.lock), so results match between host and Airflow.
ARG AIRFLOW_VERSION=3.3.2
FROM apache/airflow:${AIRFLOW_VERSION}-python3.11
ARG AIRFLOW_VERSION
ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV MLFLOW_DISABLE_AGENT_HINT=1 TICKETFLOW_ROOT=/opt/ticketflow PYTHONPATH=/opt/ticketflow/src

COPY --chown=airflow:root pyproject.toml /tmp/ticketflow/
COPY --chown=airflow:root src /tmp/ticketflow/src
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" "torch==${TORCH_VERSION}" "/tmp/ticketflow[pipeline]"
