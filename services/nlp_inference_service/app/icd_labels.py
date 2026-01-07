ICD_CODES = [
    "I20.9",
    "E11.9",
    "I10",
    "J45.909",
]

ICD_TO_INDEX = {code: i for i, code in enumerate(ICD_CODES)}
INDEX_TO_ICD = {i: code for code, i in ICD_TO_INDEX.items()}
NUM_LABELS = len(ICD_CODES)