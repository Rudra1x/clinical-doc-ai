import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

from .icd_labels import NUM_LABELS

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
MODEL_PATH = Path(__file__).parent.parent / "models" / "biobert_icd_weights.pt"


@lru_cache()
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"❌ Model weights not found at {MODEL_PATH}.\n"
            f"Please run training first."
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Classifier weight mean:", model.classifier.weight.mean().item())
    return model, tokenizer