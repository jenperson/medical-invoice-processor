import asyncio
import logging
import os
from functools import lru_cache
from datetime import timedelta
from typing import Optional

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, create_model

from shared.extraction_fields import COMMON_FIELDS, DOCUMENT_CATEGORIES, SPECIFIC_FIELDS, build_extraction_prompt

load_dotenv(override=True)

for name in ("mistralai_workflows", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)


class ManualCategorySignal(BaseModel):
    category: str


class DocumentClassification(BaseModel):
    category: str = Field(description=f"One of: {', '.join(DOCUMENT_CATEGORIES)}")
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


def extract_text_from_delta_content(delta_content: object) -> str:
    if isinstance(delta_content, str):
        return delta_content
    if not isinstance(delta_content, list):
        return ""

    parts: list[str] = []
    for chunk in delta_content:
        if isinstance(chunk, dict):
            if chunk.get("type") == "text" and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
            continue

        chunk_type = getattr(chunk, "type", None)
        chunk_text = getattr(chunk, "text", None)
        if chunk_type == "text" and isinstance(chunk_text, str):
            parts.append(chunk_text)

    return "".join(parts)


def build_classification_messages(ocr_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert in medical document classification. "
                "Return only valid JSON that matches the schema."
            ),
        },
        {
            "role": "user",
            "content": (
                "Classify this OCR document into one category from:\n"
                + "\n".join(f"- {c}" for c in DOCUMENT_CATEGORIES)
                + "\n\nOCR DOCUMENT:\n"
                + ocr_text[:8000]
            ),
        },
    ]


def build_classification_explanation_messages(ocr_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert in medical document classification. "
                "Provide a concise plain-language explanation of why this document fits a category."
            ),
        },
        {
            "role": "user",
            "content": (
                "Briefly explain your classification rationale in 2-3 sentences. "
                "Do not output JSON.\n\n"
                "Candidate categories:\n"
                + "\n".join(f"- {c}" for c in DOCUMENT_CATEGORIES)
                + "\n\nOCR DOCUMENT:\n"
                + ocr_text[:8000]
            ),
        },
    ]


@lru_cache(maxsize=None)
def get_extraction_output_model(category: str) -> type[BaseModel]:
    common_model = create_model(
        "CommonExtractionFields",
        __config__=ConfigDict(extra="forbid"),
        **{key: (Optional[str], None) for key, _ in COMMON_FIELDS},
    )
    specific_model = create_model(
        f"SpecificExtractionFields_{category}",
        __config__=ConfigDict(extra="forbid"),
        **{key: (Optional[str], None) for key, _ in SPECIFIC_FIELDS.get(category, [])},
    )
    return create_model(
        f"ExtractionOutput_{category}",
        __config__=ConfigDict(extra="forbid"),
        common=(common_model, ...),
        specific=(specific_model, ...),
    )


# ── Activities ────────────────────────────────────────────────────────────────

@workflows.activity(start_to_close_timeout=timedelta(minutes=5), retry_policy_max_attempts=2)
async def ocr_pdf(file_id: str, filename: str) -> str:
    client = workflows_mistralai.get_mistral_client()
    signed_url = await client.files.get_signed_url_async(file_id=file_id)
    ocr_response = await client.ocr.process_async(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
            "document_name": filename,
        },
    )
    return "\n\n".join(page.markdown for page in ocr_response.pages)


@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2)
async def classify_document(ocr_text: str) -> dict:
    client = workflows_mistralai.get_mistral_client()
    model = os.environ.get("MISTRAL_CLASSIFIER_MODEL", "mistral-medium-latest")

    streamed_explanation = ""
    published_explanation_len = 0
    async with workflows.task_from(
        state={"explanation": ""},
        type="classification_explanation",
    ) as explanation_task:
        stream = await client.chat.stream_async(
            model=model,
            temperature=0.1,
            messages=build_classification_explanation_messages(ocr_text),
        )
        async with stream:
            async for chunk in stream:
                if not chunk.data.choices:
                    continue
                delta = chunk.data.choices[0].delta
                delta_text = extract_text_from_delta_content(delta.content)
                if delta_text:
                    streamed_explanation += delta_text
                    # Keep updates frequent enough to feel live in the UI.
                    if len(streamed_explanation) - published_explanation_len >= 8:
                        await explanation_task.update_state({"explanation": streamed_explanation})
                        published_explanation_len = len(streamed_explanation)

        if streamed_explanation and len(streamed_explanation) != published_explanation_len:
            await explanation_task.update_state({"explanation": streamed_explanation})

    response = await client.chat.parse_async(
        response_format=DocumentClassification,
        model=model,
        temperature=0.1,
        messages=build_classification_messages(ocr_text),
    )
    parsed = response.choices[0].message.parsed if response.choices and response.choices[0].message else None
    if parsed is None:
        raise RuntimeError("Classification response could not be parsed.")

    classification = parsed.model_dump(mode="json")
    if streamed_explanation.strip():
        classification["explanation"] = streamed_explanation.strip()
    return classification


@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2)
async def extract_patient_info(ocr_text: str, category: str) -> dict:
    client = workflows_mistralai.get_mistral_client()
    model = os.environ.get("MISTRAL_EXTRACTOR_MODEL", "mistral-medium-latest")
    extraction_model = get_extraction_output_model(category)
    response = await client.chat.parse_async(
        response_format=extraction_model,
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract administrative and medical information from OCR text. "
                    "Return only valid JSON that matches the schema. "
                    "If a field is missing, set it to null."
                ),
            },
            {
                "role": "user",
                "content": build_extraction_prompt(ocr_text, category),
            },
        ],
    )
    parsed = response.choices[0].message.parsed if response.choices and response.choices[0].message else None
    if parsed is None:
        raise RuntimeError("Extraction response could not be parsed.")
    return parsed.model_dump(mode="json")


# ── Workflow ──────────────────────────────────────────────────────────────────

@workflows.workflow.define(name="pdf_ocr_workflow")
class PdfOcrWorkflow(workflows.InteractiveWorkflow):
    def __init__(self):
        self.steps = {
            "ocr": {"status": "pending", "result": None},
            "classify": {"status": "pending", "result": None},
            "extract": {"status": "pending", "result": None},
        }
        self._todo_runtime: dict[str, workflows_mistralai.TodoListItem] = {}
        self._manual_category = None

    @workflows.workflow.query(name="get_steps")
    def get_steps(self) -> dict:
        return self.steps

    @workflows.workflow.query(name="get_todo_list")
    def get_todo_list(self) -> dict:
        defs = [
            ("ocr", "Extract OCR text", "Read the uploaded PDF and extract text content."),
            ("classify", "Classify document type", "Predict the medical document category with confidence."),
            ("extract", "Extract structured fields", "Extract patient and document-specific fields."),
        ]
        items = []
        for item_id, title, description in defs:
            item = self._todo_runtime.get(item_id)
            status = getattr(item, "status", "todo") if item else "todo"
            if hasattr(status, "value"):
                status = status.value
            items.append(
                {
                    "id": item_id,
                    "title": title,
                    "description": description,
                    "status": status,
                }
            )
        return {"items": items}

    @workflows.workflow.signal(name="manual_category")
    async def manual_category_signal(self, payload: ManualCategorySignal) -> None:
        self._manual_category = payload.category

    @workflows.workflow.entrypoint
    async def run(self, file_id: str, filename: str, confidence_threshold: float = 0.9, is_batch_mode: bool = False) -> workflows_mistralai.ChatAssistantWorkflowOutput:
        ocr_item = workflows_mistralai.TodoListItem(
            title="Extract OCR text",
            description="Read the uploaded PDF and extract text content.",
        )
        classify_item = workflows_mistralai.TodoListItem(
            title="Classify document type",
            description="Predict the medical document category with confidence.",
        )
        extract_item = workflows_mistralai.TodoListItem(
            title="Extract structured fields",
            description="Extract patient and document-specific fields.",
        )
        self._todo_runtime = {
            "ocr": ocr_item,
            "classify": classify_item,
            "extract": extract_item,
        }

        async with workflows_mistralai.TodoList(items=[ocr_item, classify_item, extract_item]):
            self.steps["ocr"]["status"] = "running"
            async with ocr_item:
                ocr_text = await ocr_pdf(file_id, filename)
            self.steps["ocr"] = {"status": "done", "result": ocr_text}

            self.steps["classify"]["status"] = "running"
            async with classify_item:
                classification = await classify_document(ocr_text)

                if classification.get("confidence", 0.0) < confidence_threshold:
                    self.steps["classify"] = {"status": "waiting_human", "result": classification}
                    await workflows.workflow.wait_condition(lambda: self._manual_category is not None)
                    classification["category"] = self._manual_category
                    classification["confidence"] = 1.0
                    classification["explanation"] = f"Manually selected category: {self._manual_category}"

            self.steps["classify"] = {"status": "done", "result": classification}

            self.steps["extract"]["status"] = "running"
            async with extract_item:
                patient_info = await extract_patient_info(ocr_text, classification["category"])
            self.steps["extract"] = {"status": "done", "result": patient_info}

        return workflows_mistralai.ChatAssistantWorkflowOutput(
            content=[workflows_mistralai.TextOutput(text=f"Processing complete for {filename}.")],
            structuredContent={
                "filename": filename,
                "ocr_text": ocr_text,
                "classification": classification,
                "patient_info": patient_info,
            },
        )
        
async def main() -> None:
    print("Worker ready — waiting for tasks...\n")
    await workflows.run_worker([PdfOcrWorkflow])


if __name__ == "__main__":
    asyncio.run(main())
