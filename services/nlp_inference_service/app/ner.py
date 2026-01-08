import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
MODEL_PATH = Path(__file__).parent.parent / "models" / "biobert_ner_weights.pt"

LABELS = [
    "O",
    "B-DISEASE",
    "I-DISEASE",
    "B-MEDICATION",
    "I-MEDICATION",
    "B-SYMPTOM",
    "I-SYMPTOM",
]


def load_ner_model():
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def run_ner(text: str):
    model, tokenizer = load_ner_model()

    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]

    with torch.no_grad():
        logits = model(**encoding).logits

    predictions = torch.argmax(logits, dim=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    entities = {}
    current_entity = []
    current_label = None

    for token, label_id, mask in zip(tokens, predictions, attention_mask):
        if mask.item() == 0:
            continue

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        label = LABELS[label_id]

        if label == "O":
            if current_entity:
                entity_text = "".join(current_entity).replace("##", "")
                entities.setdefault(current_label, []).append(entity_text)
                current_entity = []
                current_label = None
            continue

        tag, entity_type = label.split("-", 1)

        if tag == "B":
            if current_entity:
                entity_text = "".join(current_entity).replace("##", "")
                entities.setdefault(current_label, []).append(entity_text)

            current_entity = [token]
            current_label = entity_type

        elif tag == "I" and current_label == entity_type:
            current_entity.append(token)

        else:
            if current_entity:
                entity_text = "".join(current_entity).replace("##", "")
                entities.setdefault(current_label, []).append(entity_text)

            current_entity = []
            current_label = None

    # Flush last entity
    if current_entity:
        entity_text = "".join(current_entity).replace("##", "")
        entities.setdefault(current_label, []).append(entity_text)

    return entities