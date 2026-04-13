"""
Run once to create the classifier and extractor agents, then store their IDs in .env.

    cp .env.example .env        # add your MISTRAL_API_KEY
    uv run python create_agents.py
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

from mistralai import Mistral, CompletionArgs, ResponseFormat, JSONSchema
from pydantic import BaseModel, Field
from typing import Optional


DOCUMENT_CATEGORIES = [
    "ordonnance",
    "facture_soin",
    "compte_rendu_hospitalisation",
    "analyse_biologique",
    "imagerie_medicale",
    "certificat_medical",
    "mutuelle_remboursement",
    "securite_sociale_remboursement",
    "compte_rendu_consultation",
    "consentement_eclaire",
    "autre",
]

CLASSIFIER_INSTRUCTIONS = f"""Tu es un expert en classification de documents médicaux.
Tu analyses le contenu OCR d'un document et tu le classes dans l'une des catégories suivantes :

{chr(10).join(f"- {c}" for c in DOCUMENT_CATEGORIES)}

Tu réponds UNIQUEMENT en JSON valide selon le schéma demandé.
Sois précis dans ton explication et justifie ta classification."""


class DocumentClassification(BaseModel):
    category: str = Field(description=f"Une des catégories : {', '.join(DOCUMENT_CATEGORIES)}")
    confidence: float = Field(ge=0.0, le=1.0, description="Niveau de confiance entre 0 et 1")
    explanation: str = Field(description="Explication courte de la classification")


EXTRACTOR_INSTRUCTIONS = """Tu es un expert en extraction d'informations administratives et médicales.
Tu analyses le contenu OCR d'un document médical et tu extrais les informations demandées.

Règles :
- Si une information n'est pas présente dans le document, retourne null pour ce champ.
- Ne devine jamais une information, extrait uniquement ce qui est explicitement écrit.
- Tu réponds UNIQUEMENT en JSON valide selon le schéma demandé."""


class PatientInfo(BaseModel):
    nom_complet: Optional[str] = Field(None, description="Nom complet du patient (nom + prénom)")
    adresse_patient: Optional[str] = Field(None, description="Adresse complète du patient")
    numero_secu: Optional[str] = Field(None, description="Numéro de sécurité sociale")


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
    env_path = os.path.join(os.path.dirname(__file__), ".env")

    print("Creating classifier agent...")
    classifier = client.beta.agents.create(
        model="mistral-medium-latest",
        name="medical-document-classifier",
        description="Classifie les documents médicaux en catégories prédéfinies avec un niveau de confiance.",
        instructions=CLASSIFIER_INSTRUCTIONS,
        completion_args=CompletionArgs(
            temperature=0.1,
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=JSONSchema(
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
        description="Extrait les informations d'identification du patient depuis un document médical.",
        instructions=EXTRACTOR_INSTRUCTIONS,
        completion_args=CompletionArgs(
            temperature=0.0,
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=JSONSchema(
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
