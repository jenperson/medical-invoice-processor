"""
Champs à extraire par type de document.
Importé par workflow.py et app.py.
"""

COMMON_FIELDS: list[tuple[str, str]] = [
    ("nom_complet", "Nom complet du patient (nom + prénom)"),
    ("adresse_patient", "Adresse complète du patient"),
    ("numero_secu", "Numéro de sécurité sociale du patient"),
]

SPECIFIC_FIELDS: dict[str, list[tuple[str, str]]] = {
    "ordonnance": [
        ("medecin_nom", "Nom du médecin prescripteur"),
        ("medecin_adresse", "Adresse du médecin prescripteur"),
        ("medecin_rpps", "Numéro RPPS du médecin"),
        ("medicaments", "Liste des médicaments prescrits (séparés par des virgules)"),
        ("date_prescription", "Date de la prescription"),
    ],
    "facture_soin": [
        ("professionnel_nom", "Nom du professionnel de santé"),
        ("professionnel_adresse", "Adresse du professionnel de santé"),
        ("montant_facture", "Montant total facturé (avec devise)"),
        ("actes", "Actes ou prestations facturés"),
        ("date_soin", "Date du soin"),
    ],
    "compte_rendu_hospitalisation": [
        ("etablissement", "Nom de l'établissement hospitalier"),
        ("service", "Service ou unité de soin"),
        ("medecin_responsable", "Médecin responsable"),
        ("date_entree", "Date d'entrée"),
        ("date_sortie", "Date de sortie"),
        ("diagnostic_principal", "Diagnostic principal"),
    ],
    "analyse_biologique": [
        ("laboratoire", "Nom du laboratoire"),
        ("medecin_prescripteur", "Médecin prescripteur"),
        ("date_prelevement", "Date du prélèvement"),
        ("analyses", "Analyses effectuées"),
        ("resultats_anormaux", "Résultats anormaux signalés"),
    ],
    "imagerie_medicale": [
        ("etablissement", "Établissement ou cabinet de radiologie"),
        ("medecin_radiologue", "Nom du radiologue"),
        ("type_examen", "Type d'examen (IRM, scanner, radio…)"),
        ("region_anatomique", "Région anatomique examinée"),
        ("date_examen", "Date de l'examen"),
        ("conclusion", "Conclusion ou résultat principal"),
    ],
    "certificat_medical": [
        ("medecin_nom", "Nom du médecin signataire"),
        ("medecin_adresse", "Adresse du médecin"),
        ("objet_certificat", "Objet du certificat"),
        ("date_certificat", "Date du certificat"),
        ("duree_arret", "Durée d'arrêt de travail si mentionnée"),
    ],
    "mutuelle_remboursement": [
        ("mutuelle_nom", "Nom de la mutuelle"),
        ("numero_adherent", "Numéro d'adhérent"),
        ("montant_rembourse", "Montant remboursé par la mutuelle"),
        ("date_remboursement", "Date du remboursement"),
        ("actes_rembourses", "Actes remboursés"),
    ],
    "securite_sociale_remboursement": [
        ("caisse_nom", "Nom de la caisse d'assurance maladie"),
        ("montant_rembourse", "Montant remboursé par la Sécu"),
        ("taux_remboursement", "Taux de remboursement appliqué"),
        ("date_remboursement", "Date du remboursement"),
        ("actes_rembourses", "Actes remboursés"),
    ],
    "compte_rendu_consultation": [
        ("medecin_nom", "Nom du médecin consultant"),
        ("medecin_specialite", "Spécialité du médecin"),
        ("date_consultation", "Date de la consultation"),
        ("motif", "Motif de la consultation"),
        ("diagnostic", "Diagnostic posé"),
        ("traitement_prescrit", "Traitement ou suivi prescrit"),
    ],
    "consentement_eclaire": [
        ("medecin_nom", "Nom du médecin"),
        ("etablissement", "Établissement"),
        ("acte_concerne", "Acte médical concerné"),
        ("date_signature", "Date de signature"),
    ],
    "autre": [],
}


def build_extraction_prompt(ocr_text: str, category: str) -> str:
    specific = SPECIFIC_FIELDS.get(category, [])

    common_lines = "\n".join(f'  - "{key}": {desc}' for key, desc in COMMON_FIELDS)
    specific_lines = (
        "\n".join(f'  - "{key}": {desc}' for key, desc in specific)
        if specific else '  (aucun champ spécifique pour ce type de document)'
    )
    specific_json_example = (
        "{" + ", ".join(f'"{k}": "..."' for k, _ in specific) + "}"
        if specific else "{}"
    )

    return f"""Type de document : {category}

Extrais les informations suivantes du document OCR ci-dessous.

CHAMPS COMMUNS (toujours présents dans "common") :
{common_lines}

CHAMPS SPÉCIFIQUES pour "{category}" (dans "specific") :
{specific_lines}

Réponds UNIQUEMENT avec un JSON valide de cette forme :
{{
  "common": {{
    "nom_complet": "...",
    "adresse_patient": "...",
    "numero_secu": "..."
  }},
  "specific": {specific_json_example}
}}

Si une information est absente du document, mets null pour la valeur correspondante.

DOCUMENT :
{ocr_text[:8000]}"""
