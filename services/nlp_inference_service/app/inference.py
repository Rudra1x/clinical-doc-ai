import torch
from .model import load_model
from .icd_labels import INDEX_TO_ICD
from .explain import compute_token_attributions

THRESHOLD = 0.5


def run_inference(text: str, explain: bool = False):
    """
    Run ICD prediction.
    Explainability is OPTIONAL and only computed for top-1 prediction.
    """

    model, tokenizer = load_model()

    # Encode input
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

    # Forward pass (FAST)
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze(0)

    # ICD predictions
    icd_predictions = [
        {
            "code": INDEX_TO_ICD[idx],
            "description": "Clinical ICD Code",
            "confidence": round(prob.item(), 3),
        }
        for idx, prob in enumerate(probs)
        if prob.item() >= THRESHOLD
    ]

    # Explainability (OPTIONAL & TOP-1 ONLY)
    explanations = {}

    if explain and len(icd_predictions) > 0:
        top_idx = torch.argmax(probs).item()
        top_code = INDEX_TO_ICD[top_idx]

        tokens, scores = compute_token_attributions(
            model=model,
            tokenizer=tokenizer,
            text=text,
            target_label_idx=top_idx,
        )

        explanations[top_code] = [
            {"token": token, "importance": float(score)}
            for token, score in zip(tokens, scores)
            if token not in ["[PAD]", "[CLS]", "[SEP]"]
        ]

    # Summary placeholder
    summary = "Automated clinical summary generation coming next."

    return summary, icd_predictions, explanations