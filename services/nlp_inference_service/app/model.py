import torch
from functools import lru_cache
from transformers import AutoTokenizer
from pathlib import Path

from .icd_labels import NUM_LABELS

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
MODEL_PATH = Path(__file__).parent.parent / "models" / "biobert_icd.ckpt"

@lru_cache()
def load_model():
    from pytorch_lightning import LightningModule
    from transformers import AutoModelForSequenceClassification

    class BioBERTICDModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=NUM_LABELS,
                problem_type="multi_label_classification"
            )

        def forward(self, **inputs):
            return self.model(**inputs)

    model = BioBERTICDModel.load_from_checkpoint(
        checkpoint_path=str(MODEL_PATH),
        map_location="cpu"
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer