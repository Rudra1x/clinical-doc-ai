from .model import load_model

def run_inference(text: str):
    model = load_model()

    # Placeholder outputs (replace later)
    summary = "Patient admitted with chest pain. Stable at discharge."

    icd_predictions = [
        {
            "code": "I20.9",
            "description": "Angina pectoris, unspecified",
            "confidence": 0.91
        }
    ]

    entities = {
        "DISEASE": ["chest pain"],
        "MEDICATION": ["aspirin"]
    }

    return summary, icd_predictions, entities