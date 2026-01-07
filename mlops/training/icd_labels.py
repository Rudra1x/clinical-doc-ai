ICD_CODES = [
    "I20.9",   # Angina
    "E11.9",   # Type 2 diabetes
    "I10",     # Hypertension
    "J45.909", # Asthma
]

ICD_TO_INDEX = {code: i for i, code in enumerate(ICD_CODES)}
INDEX_TO_ICD = {i: code for code, i in ICD_TO_INDEX.items()}
NUM_LABELS = len(ICD_CODES)