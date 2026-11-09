# Every target is a thin wrapper over `uv run ticketflow ...`, so on Windows without make
# the same commands can be run directly (see README, "Команды без make").
export MLFLOW_DISABLE_AGENT_HINT=1
STACK_MLFLOW = MLFLOW_TRACKING_URI=http://localhost:5000
ADMIN_TOKEN ?= change-me

.PHONY: install data simulate benchmark test lint up bootstrap replay rollback down

install:            ## create the venv (Python 3.11, CUDA torch on Windows)
	uv sync --extra pipeline

data:               ## download + prepare data through DVC
	uv run dvc repro prepare

simulate: data      ## six months of traffic through the full lifecycle (~5 min on RTX 4060 Ti)
	uv run ticketflow simulate

benchmark:          ## PyTorch vs ONNX vs ONNX int8 on CPU (model retrained in `simulate`, months 4-5)
	uv run ticketflow benchmark

test:
	uv run pytest -q

lint:
	uv run ruff check src tests dags

up:                 ## start MLflow, Postgres, Airflow, Prometheus, Grafana (service waits for a model)
	docker compose up -d --build postgres mlflow airflow prometheus grafana

bootstrap:          ## train the first model on the host GPU, register and promote it, start the service
	$(STACK_MLFLOW) uv run ticketflow train
	docker compose up -d service

replay:             ## stream tickets into the service from day 85 (drift starts at day 90)
	uv run ticketflow replay --start-day 85 --seconds-per-day 10

rollback:           ## production <- previous, then make the service reload; prints wall time
	$(STACK_MLFLOW) uv run ticketflow rollback --reload-url http://localhost:8000 --token $(ADMIN_TOKEN)

down:
	docker compose down
