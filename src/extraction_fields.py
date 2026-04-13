"""
Fields to extract by document type.
Imported by workflow.py and app.py.
"""

COMMON_FIELDS: list[tuple[str, str]] = [
    ("full_name", "Full name of the patient (last name + first name)"),
    ("patient_address", "Full address of the patient"),
    ("social_security_number", "Social security number of the patient"),
]

SPECIFIC_FIELDS: dict[str, list[tuple[str, str]]] = {
    "prescription": [
        ("doctor_name", "Name of the doctor who made the prescription"),
        ("doctor_address", "Address of the doctor who made the prescription"),
        ("doctor_rpps", "RPPS number of the doctor"),
        ("medications", "List of prescribed medications (separated by commas)"),
        ("prescription_date", "Date of the prescription"),
    ],
    "medical_bill": [
        ("healthcare_professional_name", "Name of the healthcare professional"),
        ("healthcare_professional_address", "Address of the healthcare professional"),
        ("bill_amount", "Total billed amount (with currency)"),
        ("services", "Billed services or procedures"),
        ("care_date", "Date of care"),
    ],
    "hospitalization_report": [
        ("hospital_name", "Name of the hospital"),
        ("department", "Department or care unit"),
        ("responsible_doctor", "Responsible doctor"),
        ("admission_date", "Date of admission"),
        ("discharge_date", "Date of discharge"),
        ("primary_diagnosis", "Primary diagnosis"),
    ],
    "biological_analysis": [
        ("laboratory", "Name of the laboratory"),
        ("prescribing_doctor", "Prescribing doctor"),
        ("sample_date", "Date of the sample"),
        ("analyses", "Performed analyses"),
        ("abnormal_results", "Reported abnormal results"),
    ],
    "medical_imaging": [
        ("facility", "Name of the imaging facility"),
        ("radiologist_name", "Name of the radiologist"),
        ("exam_type", "Type of exam (MRI, CT, X-ray, etc.)"),
        ("anatomical_region", "Anatomical region examined"),
        ("exam_date", "Date of the exam"),
        ("conclusion", "Conclusion or main finding"),
    ],
    "medical_certificate": [
        ("doctor_name", "Name of the signing doctor"),
        ("doctor_address", "Address of the doctor"),
        ("certificate_subject", "Subject of the certificate"),
        ("certificate_date", "Date of the certificate"),
        ("sick_leave_duration", "Duration of sick leave if mentioned"),
    ],
    "mutual_reimbursement": [
        ("mutual_name", "Name of the mutual insurance"),
        ("member_number", "Member number"),
        ("reimbursement_amount", "Amount reimbursed by the mutual insurance"),
        ("reimbursement_date", "Reimbursement date"),
        ("reimbursed_acts", "Reimbursed acts"),
    ],
    "social_security_reimbursement": [
        ("fund_name", "Name of the social security fund"),
        ("reimbursement_amount", "Amount reimbursed by social security"),
        ("reimbursement_rate", "Applied reimbursement rate"),
        ("reimbursement_date", "Reimbursement date"),
        ("reimbursed_acts", "Reimbursed acts"),
    ],
    "consultation_report": [
        ("doctor_name", "Name of the consulting doctor"),
        ("doctor_specialty", "Specialty of the doctor"),
        ("consultation_date", "Date of the consultation"),
        ("reason", "Reason for the consultation"),
        ("diagnosis", "Diagnosis"),
        ("prescribed_treatment", "Prescribed treatment or follow-up"),
    ],
    "informed_consent": [
        ("doctor_name", "Name of the doctor"),
        ("facility", "Facility"),
        ("medical_procedure", "Medical procedure concerned"),
        ("signature_date", "Date of signature"),
    ],
    "other": [],
}


def build_extraction_prompt(ocr_text: str, category: str) -> str:
    specific = SPECIFIC_FIELDS.get(category, [])

    common_lines = "\n".join(f'  - "{key}": {desc}' for key, desc in COMMON_FIELDS)
    specific_lines = (
        "\n".join(f'  - "{key}": {desc}' for key, desc in specific)
        if specific else '  (no specific fields for this type of document)'
    )
    specific_json_example = (
        "{" + ", ".join(f'"{k}": "..."' for k, _ in specific) + "}"
        if specific else "{}"
    )

    return f"""Document type: {category}

Extract the following information from the OCR document below.

COMMON FIELDS (always present in "common"):
{common_lines}

SPECIFIC FIELDS for "{category}" (in "specific"):
{specific_lines}

Respond ONLY with a valid JSON in this format:
{{
  "common": {{
    "full_name": "...",
    "patient_address": "...",
    "social_security_number": "..."
  }},
  "specific": {specific_json_example}
}}

If any information is missing from the document, set the corresponding value to null.

DOCUMENT:
{ocr_text[:8000]}"""
