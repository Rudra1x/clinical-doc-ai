import torch
from functools import lru_cache

@lru_cache()
def load_model():
    """
    Load BioBERT / ICD model once per container.
    Replace this with real model loading later.
    """
    model = "dummy-biobert-model"
    return model