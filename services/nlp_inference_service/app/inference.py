import torch
from .model import load_model
from .icd_labels import INDEX_TO_ICD
from .explain import compute_token_attributions
from .ner import run_ner
from .context import has_diagnostic_context, symptom_based_icd

THRESHOLD = 0.5


def run_inference(text: str, explain: bool = False):
    """
    Context-aware inference:
    - Symptom-only input  → Symptom ICD codes (R / J codes)
    - Diagnostic context → Disease ICD prediction (ML)
    """

    # Step 1: Run NER first (always)
    ner_entities = run_ner(text)

    # Step 2: Context gate
    if not has_diagnostic_context(text, ner_entities):
        # Symptom-only case (clinically correct behavior)
        symptom_codes = symptom_based_icd(text)

        return (
            "Symptoms detected. Diagnostic confirmation required for ICD-10 disease coding.",
            symptom_codes,
            {},               # no explainability
            ner_entities,
        )

    # Step 3: Disease ICD prediction (ML path)

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

    icd_predictions = [
        {
            "code": INDEX_TO_ICD[idx],
            "description": "Clinical ICD Code",
            "confidence": round(prob.item(), 3),
        }
        for idx, prob in enumerate(probs)
        if prob.item() >= THRESHOLD
    ]

    # Step 4: Optional explainability (TOP-1 only)

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

    summary = "ICD-10 disease coding performed using diagnostic clinical context."

    return summary, icd_predictions, explanations, ner_entities