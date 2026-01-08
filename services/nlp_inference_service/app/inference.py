import torch
from .model import load_model
from .icd_labels import INDEX_TO_ICD
from .explain import compute_token_attributions

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
    explanations = {}

    for idx, prob in enumerate(probs):
        if prob.item() >= THRESHOLD:
            code = INDEX_TO_ICD[idx]

            tokens, scores = compute_token_attributions(
                model, tokenizer, text, idx
            )

            explanations[code] = [
                {"token": t, "importance": float(s)}
                for t, s in zip(tokens, scores)
                if t not in ["[PAD]", "[CLS]", "[SEP]"]
            ]

            icd_predictions.append(
                {
                    "code": code,
                    "description": "Clinical ICD Code",
                    "confidence": round(prob.item(), 3),
                }
            )

    summary = "Automated clinical summary generation coming next."

    return summary, icd_predictions, explanations