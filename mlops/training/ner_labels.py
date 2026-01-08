NER_LABELS = [
    "O",
    "B-DISEASE",
    "I-DISEASE",
    "B-MEDICATION",
    "I-MEDICATION",
    "B-SYMPTOM",
    "I-SYMPTOM",
]

LABEL2ID = {label: i for i, label in enumerate(NER_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS = len(NER_LABELS)