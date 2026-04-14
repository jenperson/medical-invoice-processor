.PHONY: start-worker execute installdeps

## Install dependencies
installdeps:
	uv sync

## Auto-discover all workflows and start the worker (with file-watch auto-reload)
start-worker:
	uv run python src/dev_worker.py

## Trigger a workflow execution
## Usage: make execute workflow=hello-world input='{"name": "World"}'
execute:
	uv run python src/workflows/start.py $(if $(workflow),--workflow $(workflow),) $(if $(input),--input '$(input)',)

## Create agents (if they don't exist already). Only needs to run once.
create-agents:
	PYTHONPATH=src uv run python src/entrypoints/create_agents.py

## Start the Streamlit app
streamlit:
	PYTHONPATH=src uv run streamlit run src/entrypoints/app.py

## Run load test
load-test:
	uv run python src/entrypoints/load_test.py $(if $(n),$(n),)