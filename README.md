# OCR-driven document processing for healthcare

A [Mistral Workflows](https://docs.mistral.ai/workflows/getting-started/introduction) project.

This project shows how to use Mistral Workflows to process documents in healthcare.

## Setup

Run the following command to install the required project dependencies:

```bash
make install-deps
```

## Commands

### Create the agents 

This workflow makes use of two agents: a medical document classifier, and a medical patient extractor. The following command creates the agents, if they don't exist already. This process only needs to run once.

```bash
make create-agents
```

This command also adds the agent IDs, a deployment ID, and a build ID to your `.env` file.

### Register workflows in AI Studio

Use the following command to auto-discoves all workflow classes in `src/workflows/`, register them with AI Studio, and starts polling for executions. The task queue is set to your hostname:

```bash
make start-worker
```

### Execute the workflow

In a separate terminal, start the Streamlit app to view the document extraction UI:

```bash
make streamlit
```

Upload one of the provided PDFs and click "Start Workflow".

You can monitor your workflow's progress and view the extracted data in [AI Studio](https://console.mistral.ai/build/workflows/).

### Load test

The following command runs a load test that triggers the workflow 10 times using the provided input documents:

```bash
make load-test
```

You can also provide a custom amount of times to trigger the workflow:

```bash
make load-test n=5
```

## Development

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check --fix .
```

## Clean up

When you're done, stop the Streamlit app and the worker using `Ctrl+C`.
