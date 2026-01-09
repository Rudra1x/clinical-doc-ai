DIAGNOSIS_KEYWORDS = [
    "diagnosed",
    "diagnosis",
    "assessment",
    "impression",
    "confirmed",
    "acute",
    "chronic",
]

SYMPTOM_ICD_MAP = {
    "cough": ("R05", "Cough"),
    "pain": ("R52", "Pain, unspecified"),
    "throat": ("J02.9", "Acute pharyngitis, unspecified"),
    "fever": ("R50.9", "Fever, unspecified"),
    "chest pain": ("R07.9", "Chest pain, unspecified"),
}


def has_diagnostic_context(text: str, ner_entities: dict) -> bool:
    text_lower = text.lower()

    # Keyword-based check
    for kw in DIAGNOSIS_KEYWORDS:
        if kw in text_lower:
            return True

    # If disease entity exists, assume diagnostic intent
    if any(
        label.lower().startswith("disease")
        for label in ner_entities.keys()
    ):
        return True

    return False


def symptom_based_icd(text: str):
    text_lower = text.lower()
    codes = []

    for symptom, (code, desc) in SYMPTOM_ICD_MAP.items():
        if symptom in text_lower:
            codes.append(
                {
                    "code": code,
                    "description": desc,
                    "confidence": 1.0,
                }
            )

    return codes