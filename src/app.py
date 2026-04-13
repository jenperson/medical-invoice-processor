"""
Streamlit UI for PDF OCR + Classification + Extraction Workflow.

Terminal 1 — worker:   uv run python workflow.py
Terminal 2 — UI:       uv run streamlit run app.py
"""
import asyncio
import base64
import io
import os
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

load_dotenv(override=True)

import mistralai
from mistralai.client import Mistral
from mistralai.workflows import workflow

API_KEY = os.environ["MISTRAL_API_KEY"]
BASE_URL = "https://api.mistral.ai"

CATEGORY_LABELS = {
    "ordonnance": "📋 Ordonnance",
    "facture_soin": "🧾 Facture de soin",
    "compte_rendu_hospitalisation": "🏥 Compte-rendu d'hospitalisation",
    "analyse_biologique": "🔬 Analyse biologique",
    "imagerie_medicale": "🩻 Imagerie médicale",
    "certificat_medical": "📄 Certificat médical",
    "mutuelle_remboursement": "💳 Remboursement mutuelle",
    "securite_sociale_remboursement": "🏛️ Remboursement Sécu",
    "compte_rendu_consultation": "👨‍⚕️ Compte-rendu de consultation",
    "consentement_eclaire": "✍️ Consentement éclairé",
    "autre": "❓ Autre",
}

COMMON_FIELD_LABELS = {
    "nom_complet": "Nom complet",
    "adresse_patient": "Adresse",
    "numero_secu": "N° Sécurité sociale",
}

STEPS_CONFIG = [
    ("ocr",      "🔍 OCR du document"),
    ("classify", "🏷️ Classification"),
    ("extract",  "👤 Extraction patient"),
]


class PdfOcrInput(BaseModel):
    file_id: str
    filename: str
    confidence_threshold: float = 0.9
    is_batch_mode: bool = False  # True for load tests, False for Streamlit


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def upload_pdf(pdf_bytes: bytes, filename: str) -> str:
    async with Mistral(api_key=API_KEY) as client:
        resp = await client.files.upload_async(
            file={"file_name": filename, "content": pdf_bytes, "content_type": "application/pdf"},
            purpose="ocr",
        )
    return resp.id


async def trigger_workflow(file_id: str, filename: str, confidence_threshold: float) -> str:
    execution_id = f"pdf-ocr-{uuid.uuid4().hex[:12]}"
    async with workflow(base_url=BASE_URL, api_key=API_KEY, timeout=30.0) as client:
        resp = await client.execute_workflow(
            workflow_identifier="pdf_ocr_workflow",
            input_data=PdfOcrInput(file_id=file_id, filename=filename, confidence_threshold=confidence_threshold),
            execution_id=execution_id,
        )
    return resp.execution_id


async def poll_steps(execution_id: str) -> dict:
    async with workflow(base_url=BASE_URL, api_key=API_KEY, timeout=30.0) as client:
        resp = await client.query_workflow(
            execution_id=execution_id,
            query_name="get_steps",
        )
    return resp.result or {}


async def get_execution_status(execution_id: str) -> str:
    async with workflow(base_url=BASE_URL, api_key=API_KEY, timeout=30.0) as client:
        resp = await client.get_workflow_execution(execution_id=execution_id)
    return resp.status


class ManualCategorySignal(BaseModel):
    category: str


async def send_signal(execution_id: str, category: str):
    async with workflow(base_url=BASE_URL, api_key=API_KEY, timeout=30.0) as client:
        await client.signal_workflow(
            execution_id=execution_id,
            signal_name="manual_category",
            input_data=ManualCategorySignal(category=category),
        )


# ── PDF rendering ─────────────────────────────────────────────────────────────

def get_pdf_first_page(pdf_bytes: bytes):
    """Convert first page of PDF to image."""
    if not fitz:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 1.5x zoom for clarity
        img_bytes = pix.tobytes("ppm")
        return io.BytesIO(img_bytes)
    except Exception:
        return None


# ── Step renderers ────────────────────────────────────────────────────────────

def render_step(key: str, step: dict):
    status = step.get("status", "pending")
    result = step.get("result")

    if status == "pending":
        st.markdown("⏳ En attente…")
    elif status == "running":
        st.spinner("⚙️ En cours…")
        # spinner widget needs a context manager — use a visual substitute
        st.markdown("⚙️ En cours…")
    elif status == "waiting_human":
        result = step.get("result", {})
        confidence = result.get("confidence", 0.0) if result else 0.0
        st.warning(
            f"⚠️ Confiance insuffisante ({confidence * 100:.0f}%). "
            "Choisissez la catégorie manuellement."
        )
        selected = st.selectbox(
            "Catégorie",
            options=list(CATEGORY_LABELS.keys()),
            format_func=lambda k: CATEGORY_LABELS[k],
            key="manual_category_select",
        )
        if st.button("Valider", key="manual_category_submit"):
            run_async(send_signal(st.session_state.execution_id, selected))
            st.session_state.signal_sent = True
            st.rerun()
    elif status == "done" and result is not None:
        if key == "ocr":
            with st.expander("Texte OCR brut", expanded=False):
                st.markdown(result)
        elif key == "classify":
            category  = result.get("category", "autre")
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "")
            label = CATEGORY_LABELS.get(category, f"❓ {category}")
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{label}**")
            col1.caption(explanation)
            col2.metric("Confiance", f"{confidence * 100:.0f}%")
            col2.progress(confidence)
        elif key == "extract":
            common = result.get("common", {})
            specific = result.get("specific", {})

            st.markdown("**🧍 Informations patient**")
            common_rows = [
                {"Champ": COMMON_FIELD_LABELS.get(k, k), "Valeur": v}
                for k, v in common.items() if v is not None
            ]
            if common_rows:
                st.table(common_rows)
            else:
                st.info("Aucune information commune trouvée.")

            if specific:
                st.markdown("**📋 Informations spécifiques**")
                specific_rows = [
                    {"Champ": k.replace("_", " ").capitalize(), "Valeur": v}
                    for k, v in specific.items() if v is not None
                ]
                if specific_rows:
                    st.table(specific_rows)
                else:
                    st.info("Aucune information spécifique trouvée.")


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="PDF OCR & Classification", page_icon="📄", layout="wide")
st.title("📄 PDF OCR & Classification")
st.caption("Upload a PDF → OCR → Classification → Extraction patient")

with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence_threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="En dessous de ce seuil, la classification est soumise à validation manuelle.",
    )
    st.caption(f"Seuil actuel : **{confidence_threshold * 100:.0f}%**")
    if confidence_threshold >= 1.0:
        st.info("☝️ Validation manuelle systématique")
    elif confidence_threshold == 0.0:
        st.info("✅ Validation manuelle jamais déclenchée")

# Init session state
if "execution_id" not in st.session_state:
    st.session_state.execution_id = None
if "done" not in st.session_state:
    st.session_state.done = False
if "steps" not in st.session_state:
    st.session_state.steps = {}
if "signal_sent" not in st.session_state:
    st.session_state.signal_sent = False


uploaded = st.file_uploader("Choisir un fichier PDF", type=["pdf"])

if uploaded is not None:
    st.info(f"**{uploaded.name}** — {uploaded.size / 1024:.1f} KB")

    if st.button("Lancer le workflow", type="primary"):
        # Reset state
        st.session_state.execution_id = None
        st.session_state.done = False
        st.session_state.steps = {}
        st.session_state.signal_sent = False

        pdf_bytes = uploaded.read()
        filename = uploaded.name

        with st.status("Upload du PDF…", expanded=False) as s:
            file_id = run_async(upload_pdf(pdf_bytes, filename))
            s.update(label="Upload ✓", state="complete")

        execution_id = run_async(trigger_workflow(file_id, filename, confidence_threshold))
        st.session_state.execution_id = execution_id
        st.rerun()

# If a workflow is running, show steps
if st.session_state.execution_id and not st.session_state.done:
    execution_id = st.session_state.execution_id

    # Poll
    try:
        steps = run_async(poll_steps(execution_id))
        st.session_state.steps = steps
    except Exception:
        steps = st.session_state.steps

    col_pdf, col_steps = st.columns([1, 1.2])

    with col_pdf:
        st.markdown("### 📄 Document")
        if uploaded:
            pdf_bytes = uploaded.read()
            uploaded.seek(0)
            img = get_pdf_first_page(pdf_bytes)
            if img:
                st.image(img, width='stretch')
            else:
                st.info("PyMuPDF not available for preview")

    with col_steps:
        for key, title in STEPS_CONFIG:
            st.markdown(f"### {title}")
            step = steps.get(key, {"status": "pending", "result": None})
            render_step(key, step)

    # Check if all done
    all_done = all(
        steps.get(k, {}).get("status") == "done"
        for k, _ in STEPS_CONFIG
    )

    waiting_human = any(
        steps.get(k, {}).get("status") == "waiting_human"
        for k, _ in STEPS_CONFIG
    )

    if all_done:
        st.session_state.done = True
        st.success("✅ Terminé !")
    elif waiting_human and not st.session_state.signal_sent:
        pass
    elif waiting_human and st.session_state.signal_sent:
        time.sleep(0.5)
        st.rerun()
    else:
        try:
            wf_status = run_async(get_execution_status(execution_id))
            if wf_status in ("FAILED", "CANCELED", "TERMINATED"):
                st.error(f"Workflow terminé avec statut : {wf_status}")
                st.session_state.done = True
            else:
                time.sleep(0.5)
                st.rerun()
        except Exception:
            time.sleep(0.5)
            st.rerun()

# If workflow completed, render final state
elif st.session_state.execution_id and st.session_state.done:
    steps = st.session_state.steps
    for key, title in STEPS_CONFIG:
        st.markdown(f"### {title}")
        step = steps.get(key, {"status": "pending", "result": None})
        render_step(key, step)
    st.success("✅ Terminé !")
