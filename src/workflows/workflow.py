import asyncio
import json
import logging
import os
from datetime import timedelta

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from dotenv import load_dotenv
from pydantic import BaseModel

from extraction_fields import build_extraction_prompt


for name in ("mistralai_workflows", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)

load_dotenv(override=True)


# ── Models ────────────────────────────────────────────────────────────────────

class ManualCategorySignal(BaseModel):
    category: str


# ── Activities ────────────────────────────────────────────────────────────────

@workflows.activity(name="ocr_pdf", start_to_close_timeout=timedelta(minutes=5), retry_policy_max_attempts=2)
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


@workflows.activity(name="classify_document", start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2)
async def classify_document(ocr_text: str) -> dict:
    client = workflows_mistralai.get_mistral_client()
    response = await client.beta.conversations.start_async(
        agent_id=os.environ["CLASSIFIER_AGENT_ID"],
        inputs=f"Voici le contenu OCR du document à classifier :\n\n{ocr_text[:8000]}",
    )
    return json.loads(response.outputs[-1].content)


@workflows.activity(name="extract_patient_info", start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2)
async def extract_patient_info(ocr_text: str, category: str) -> dict:
    client = workflows_mistralai.get_mistral_client()
    response = await client.beta.conversations.start_async(
        agent_id=os.environ["EXTRACTOR_AGENT_ID"],
        inputs=build_extraction_prompt(ocr_text, category),
    )
    return json.loads(response.outputs[-1].content)


# ── Workflow ──────────────────────────────────────────────────────────────────

@workflows.workflow.define(name="pdf_ocr_workflow")
class PdfOcrWorkflow:
    def __init__(self):
        self.steps = {
            "ocr": {"status": "pending", "result": None},
            "classify": {"status": "pending", "result": None},
            "extract": {"status": "pending", "result": None},
        }
        self._manual_category = None

    @workflows.workflow.query(name="get_steps")
    def get_steps(self) -> dict:
        return self.steps

    @workflows.workflow.signal(name="manual_category")
    async def manual_category_signal(self, payload: ManualCategorySignal) -> None:
        self._manual_category = payload.category

    @workflows.workflow.entrypoint
    async def run(self, file_id: str, filename: str, confidence_threshold: float = 0.9, is_batch_mode: bool = False) -> dict:
        self.steps["ocr"]["status"] = "running"
        ocr_text = await ocr_pdf(file_id, filename)
        self.steps["ocr"] = {"status": "done", "result": ocr_text}

        self.steps["classify"]["status"] = "running"
        classification = await classify_document(ocr_text)

        if classification.get("confidence", 0.0) < confidence_threshold:
            self.steps["classify"] = {"status": "waiting_human", "result": classification}
            await workflows.workflow.wait_condition(lambda: self._manual_category is not None)
            classification["category"] = self._manual_category
            classification["confidence"] = 1.0
            classification["explanation"] = f"Manually selected category: {self._manual_category}"

        self.steps["classify"] = {"status": "done", "result": classification}

        self.steps["extract"]["status"] = "running"
        patient_info = await extract_patient_info(ocr_text, classification["category"])
        self.steps["extract"] = {"status": "done", "result": patient_info}

        return {"filename": filename, "ocr_text": ocr_text, "classification": classification, "patient_info": patient_info}
        
async def main() -> None:
    print("Worker ready — waiting for tasks...\n")
    await workflows.run_worker([PdfOcrWorkflow])


if __name__ == "__main__":
    asyncio.run(main())
