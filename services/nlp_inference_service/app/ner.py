from transformers import pipeline

# Load once (cached by HF)
ner_pipeline = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    tokenizer="d4data/biomedical-ner-all",
    aggregation_strategy="simple",  # merges subwords automatically
)


def run_ner(text: str):
    """
    Run pretrained biomedical NER.
    Returns clean, human-readable entities.
    """

    results = ner_pipeline(text)

    entities = {}

    for ent in results:
        label = ent["entity_group"]  # e.g., DISEASE, CHEMICAL
        value = ent["word"]

        entities.setdefault(label, []).append(value)

    return entities