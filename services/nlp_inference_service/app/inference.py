import torch
from .model import load_model
from .icd_labels import INDEX_TO_ICD

THRESHOLD = 0.5


def run_inference(text: str):
    model, tokenizer = load_model()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze(0)

    icd_predictions = []
    for idx, prob in enumerate(probs):
        if prob.item() >= THRESHOLD:
            icd_predictions.append(
                {
                    "code": INDEX_TO_ICD[idx],
                    "description": "Clinical ICD Code",
                    "confidence": round(prob.item(), 3),
                }
            )

    summary = "Automated clinical summary generation coming next."
    entities = {"NOTE": ["NER module will be integrated next"]}

    return summary, icd_predictions, entities