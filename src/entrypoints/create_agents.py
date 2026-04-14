"""
Run once to create the classifier and extractor agents, then store their IDs in .env.
"""
import os
from datetime import date
from dotenv import load_dotenv

load_dotenv(override=True)

from mistralai.client import Mistral, models
from pydantic import BaseModel, Field
from typing import Optional

from shared.extraction_fields import DOCUMENT_CATEGORIES

CLASSIFIER_INSTRUCTIONS = f"""You are an expert in medical document classification.
You analyze the OCR content of a document and classify it into one of the following categories:

{chr(10).join(f"- {c}" for c in DOCUMENT_CATEGORIES)}

You must respond ONLY in valid JSON according to the requested schema.
Be precise in your explanation and justify your classification."""


class DocumentClassification(BaseModel):
    category: str = Field(description=f"One of the categories: {', '.join(DOCUMENT_CATEGORIES)}")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level between 0 and 1")
    explanation: str = Field(description="Short explanation of the classification")


EXTRACTOR_INSTRUCTIONS = """You are an expert in extracting administrative and medical information.
You analyze the OCR content of a medical document and extract the requested information.

Rules:
- Recognize field names whether they're in French or English, translating to the expected output keys if necessary (e.g. "nom du patient" or "patient name" should be extracted as "full_name").
- If information is not present in the document, return null for that field.
- Never guess information, only extract what is explicitly written.
- You must respond ONLY in valid JSON according to the requested schema."""


class PatientInfo(BaseModel):
    full_name: Optional[str] = Field(None, description="The complete name of the patient (first and last name)")
    patient_address: Optional[str] = Field(None, description="The complete address of the patient")
    social_security_number: Optional[str] = Field(None, description="The social security number of the patient")


def _save_to_env(env_path: str, key: str, value: str):
    with open(env_path, "r") as f:
        content = f.read()
    if key in content:
        lines = [l if not l.startswith(key) else f"{key}={value}" for l in content.splitlines()]
        new_content = "\n".join(lines) + "\n"
    else:
        new_content = content.rstrip("\n") + f"\n{key}={value}\n"
    with open(env_path, "w") as f:
        f.write(new_content)


def main():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise SystemExit("❌ Set MISTRAL_API_KEY in your .env file first.")

    client = Mistral(api_key=api_key)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(project_root, ".env")
    build_id = f"{date.today().isoformat()}.1"
    _save_to_env(env_path, "DEPLOYMENT_NAME", "invoice-processor")
    _save_to_env(env_path, "BUILD_ID", build_id)

    print("Creating classifier agent...")
    classifier = client.beta.agents.create(
        model="mistral-medium-latest",
        name="medical-document-classifier",
        description="Classify medical documents into predefined categories with a confidence level.",
        instructions=CLASSIFIER_INSTRUCTIONS,
        completion_args=models.CompletionArgs(
            temperature=0.1,
            response_format=models.ResponseFormat(
                type="json_schema",
                json_schema=models.JSONSchema(
                    name="document_classification",
                    schema=DocumentClassification.model_json_schema(),
                ),
            ),
        ),
    )
    print(f"  ✓ CLASSIFIER_AGENT_ID={classifier.id}")
    _save_to_env(env_path, "CLASSIFIER_AGENT_ID", classifier.id)

    print("Creating extractor agent...")
    extractor = client.beta.agents.create(
        model="mistral-medium-latest",
        name="medical-patient-extractor",
        description="Extract patient identification information from a medical document.",
        instructions=EXTRACTOR_INSTRUCTIONS,
        completion_args=models.CompletionArgs(
            temperature=0.0,
            response_format=models.ResponseFormat(
                type="json_schema",
                json_schema=models.JSONSchema(
                    name="patient_info",
                    schema=PatientInfo.model_json_schema(),
                ),
            ),
        ),
    )
    print(f"  ✓ EXTRACTOR_AGENT_ID={extractor.id}")
    _save_to_env(env_path, "EXTRACTOR_AGENT_ID", extractor.id)

    print("\n✅ Done. Agent IDs saved to .env.")


if __name__ == "__main__":
    main()
