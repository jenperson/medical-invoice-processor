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

from mistralai.client import Mistral
from mistralai.workflows.client import get_mistral_client

API_KEY = os.environ["MISTRAL_API_KEY"]
BASE_URL = "https://api.mistral.ai"
WORKFLOWS_CLIENT = None

CATEGORY_LABELS = {
    "prescription": "📋 Prescription",
    "medical_bill": "🧾 Medical Bill",
    "hospitalization_report": "🏥 Hospitalization Report",
    "biological_analysis": "🔬 Biological Analysis",
    "medical_imaging": "🩻 Medical Imaging",
    "medical_certificate": "📄 Medical Certificate",
    "mutual_reimbursement": "💳 Mutual Reimbursement",
    "social_security_reimbursement": "🏛️ Social Security Reimbursement",
    "consultation_report": "👨‍⚕️ Consultation Report",
    "informed_consent": "✍️ Informed Consent",
    "other": "❓ Other",
}

COMMON_FIELD_LABELS = {
    "full_name": "Full Name",
    "patient_address": "Address",
    "social_security_number": "Social Security Number",
}

STEPS_CONFIG = [
    ("ocr",      "🔍 OCR of the document"),
    ("classify", "🏷️ Classification"),
    ("extract",  "👤 Patient Extraction"),
]


class PdfOcrInput(BaseModel):
    file_id: str
    filename: str
    confidence_threshold: float = 0.9
    is_batch_mode: bool = False  # True for load tests, False for Streamlit


def get_workflows_client():
    global WORKFLOWS_CLIENT
    if WORKFLOWS_CLIENT is None:
        WORKFLOWS_CLIENT = get_mistral_client(
            server_url=BASE_URL,
            api_key=API_KEY,
        )
    return WORKFLOWS_CLIENT


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
    client = get_workflows_client()
    resp = await client.workflows.execute_workflow_async(
        workflow_identifier="pdf_ocr_workflow",
        input=PdfOcrInput(file_id=file_id, filename=filename, confidence_threshold=confidence_threshold).model_dump(mode="json"),
        execution_id=execution_id,
    )
    return resp.execution_id


async def poll_steps(execution_id: str) -> dict:
    client = get_workflows_client()
    resp = await client.workflows.executions.query_workflow_execution_async(
        execution_id=execution_id,
        name="get_steps",
    )
    return resp.result or {}


async def get_execution_status(execution_id: str) -> str:
    client = get_workflows_client()
    resp = await client.workflows.executions.get_workflow_execution_async(execution_id=execution_id)
    return resp.status


class ManualCategorySignal(BaseModel):
    category: str


async def send_signal(execution_id: str, category: str):
    client = get_workflows_client()
    await client.workflows.executions.signal_workflow_execution_async(
        execution_id=execution_id,
        name="manual_category",
        input=ManualCategorySignal(category=category).model_dump(mode="json"),
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
        st.markdown("⏳ Pending…")
    elif status == "running":
        st.spinner("⚙️ In Progress…")
        # spinner widget needs a context manager — use a visual substitute
        st.markdown("⚙️ In Progress…")
    elif status == "waiting_human":
        result = step.get("result", {})
        confidence = result.get("confidence", 0.0) if result else 0.0
        st.warning(
            f"⚠️ Insufficient confidence ({confidence * 100:.0f}%). "
            "Please choose the category manually."
        )
        selected = st.selectbox(
            "Category",
            options=list(CATEGORY_LABELS.keys()),
            format_func=lambda k: CATEGORY_LABELS[k],
            key="manual_category_select",
        )
        if st.button("Validate", key="manual_category_submit"):
            run_async(send_signal(st.session_state.execution_id, selected))
            st.session_state.signal_sent = True
            st.rerun()
    elif status == "done" and result is not None:
        if key == "ocr":
            with st.expander("Raw OCR Text", expanded=False):
                st.markdown(result)
        elif key == "classify":
            category  = result.get("category", "other")
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "")
            label = CATEGORY_LABELS.get(category, f"❓ {category}")
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{label}**")
            col1.caption(explanation)
            col2.metric("Confidence", f"{confidence * 100:.0f}%")
            col2.progress(confidence)
        elif key == "extract":
            common = result.get("common", {})
            specific = result.get("specific", {})

            st.markdown("**🧍 Patient Information**")
            common_rows = [
                {"Field": COMMON_FIELD_LABELS.get(k, k), "Value": v}
                for k, v in common.items() if v is not None
            ]
            if common_rows:
                st.table(common_rows)
            else:
                st.info("No common information found.")

            if specific:
                st.markdown("**📋 Specific Information**")
                specific_rows = [
                    {"Field": k.replace("_", " ").capitalize(), "Value": v}
                    for k, v in specific.items() if v is not None
                ]
                if specific_rows:
                    st.table(specific_rows)
                else:
                    st.info("No specific information found.")


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="PDF OCR & Classification", page_icon="📄", layout="wide")
st.title("📄 PDF OCR & Classification")
st.caption("Upload a PDF → OCR → Classification → Patient Extraction")

with st.sidebar:
    st.header("⚙️ Parameters")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Below this threshold, classification requires manual validation.",
    )
    st.caption(f"Current threshold: **{confidence_threshold * 100:.0f}%**")
    if confidence_threshold >= 1.0:
        st.info("☝️ Manual validation always required")
    elif confidence_threshold == 0.0:
        st.info("✅ Manual validation never required")

# Init session state
if "execution_id" not in st.session_state:
    st.session_state.execution_id = None
if "done" not in st.session_state:
    st.session_state.done = False
if "steps" not in st.session_state:
    st.session_state.steps = {}
if "signal_sent" not in st.session_state:
    st.session_state.signal_sent = False


uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded is not None:
    st.info(f"**{uploaded.name}** — {uploaded.size / 1024:.1f} KB")

    if st.button("Start Workflow", type="primary"):
        # Reset state
        st.session_state.execution_id = None
        st.session_state.done = False
        st.session_state.steps = {}
        st.session_state.signal_sent = False

        pdf_bytes = uploaded.read()
        filename = uploaded.name

        with st.status("Uploading PDF…", expanded=False) as s:
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
        st.success("✅ Completed!")
    elif waiting_human and not st.session_state.signal_sent:
        pass
    elif waiting_human and st.session_state.signal_sent:
        time.sleep(0.5)
        st.rerun()
    else:
        try:
            wf_status = run_async(get_execution_status(execution_id))
            if wf_status in ("FAILED", "CANCELED", "TERMINATED"):
                st.error(f"Workflow ended with status: {wf_status}")
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
    st.success("✅ Completed!")
