# invoice-processor

A [Mistral Workflows](https://docs.mistral.ai/workflows/getting-started/introduction) project.

## Setup

```bash
make installdeps
```

## Commands

### Register workflows in AI Studio

Auto-discovers all workflow classes in `src/workflows/`, registers them with AI Studio, and starts polling for executions. The task queue is set to your hostname:

```bash
make start-worker
```

###  Create agents 

Creates the agents for the ocr text classification and patient info extraction (if they don't exist already). Only needs to run once.

```bash
make create-agents
```

### Execute a workflow

In a separate terminal, start the Streamlit app to view the document extraction UI:

```bash
make streamlit
```

Upload one of the provided PDFs and click "Start Workflow".

## Development

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check --fix .
```
