# 🧠 AI-Powered Clinical Document Intelligence System

A **context-aware, explainable clinical NLP system** for automated medical document understanding, designed to **align with real-world healthcare workflows** rather than naive end-to-end prediction.

This project demonstrates how **responsible AI systems** for healthcare must combine **machine learning, rules, and domain knowledge** to produce clinically valid outputs.

---

## 📌 Problem Statement

Hospitals and healthcare providers generate large volumes of unstructured clinical text such as:

- Discharge summaries  
- Progress notes  
- Clinical assessments  

Manual processing of these documents for:
- ICD-10 coding  
- Clinical entity extraction  
- Documentation review  

is **time-consuming, error-prone, and costly**.

However, **naively applying ML models** to predict ICD codes directly from short symptom descriptions leads to **clinically incorrect and unsafe results**.

---

## 🎯 Project Objective

Build a **production-style clinical document intelligence system** that:

- Extracts meaningful clinical entities
- Predicts ICD-10 codes **only when diagnostic context is sufficient**
- Defers predictions responsibly when input is under-specified
- Provides explainability for model predictions
- Is deployable as a real-time API with a live UI

---

## 🧠 Key Design Philosophy (Very Important)

> **ICD-10 coding requires diagnostic context, not just symptoms.**

This system explicitly models that reality by introducing a **context gate** before ICD prediction.

---

## 🏗️ System Architecture

Clinical Text
↓
Biomedical NER (Pretrained Model)
↓
Context Gate
├─ Symptom-only input → Symptom ICD codes (R / J codes)
└─ Diagnostic context → ML-based ICD-10 prediction
↓
Explainability (Captum)
↓
FastAPI Inference Service
↓
Streamlit Frontend (Live Demo)


---

## 🔍 Core Features

### ✅ Context-Aware ICD-10 Coding
- Detects whether input text contains **diagnostic-level information**
- Prevents unsafe disease prediction from symptom-only inputs
- Mimics real clinical decision support systems (CDSS)

### ✅ Symptom-Based ICD Coding
- Maps symptoms to valid **ICD-10 R-codes / J-codes**
- Example:
  - Cough → `R05`
  - Throat pain → `J02.9`

### ✅ ML-Based Disease Coding (When Appropriate)
- Uses **BioBERT** for multi-label ICD prediction
- Activated only when diagnostic context is present

### ✅ Explainable AI
- Integrated Gradients (Captum)
- Token-level attribution for top ICD prediction
- Explainability is **optional and on-demand** to manage latency

### ✅ Clinical Named Entity Recognition (NER)
- Uses a **pretrained biomedical NER model**
- Extracts:
  - Symptoms
  - Diseases
  - Anatomical structures
- Clean, human-readable entity spans

### ✅ Production-Style API
- FastAPI backend
- Clear request/response schemas
- Health check endpoint
- Modular, extensible design

### ✅ Live Frontend Demo
- Streamlit-based UI
- Real-time text analysis
- ICD confidence visualization
- Entity display

---

## 🧰 Technology Stack

### 🧠 Machine Learning & NLP
- **BioBERT**
- **HuggingFace Transformers**
- **Captum (Explainability)**

### 🏥 Clinical NLP
- Biomedical NER (pretrained)
- ICD-10 coding logic
- Symptom-to-code mapping

### ⚙️ Backend & MLOps
- **FastAPI**
- **PyTorch**
- **PyTorch Lightning**
- **Weights & Biases (experiment tracking)**

### 🖥️ Frontend
- **Streamlit**

---

## 📥 Example Inputs & Outputs

### 🔹 Symptom-Only Input

**Output**
```json
{
  "summary": "Symptoms detected. Diagnostic confirmation required for ICD-10 disease coding.",
  "icd_predictions": [
    { "code": "R05", "description": "Cough", "confidence": 1.0 },
    { "code": "J02.9", "description": "Acute pharyngitis, unspecified", "confidence": 1.0 }
  ]
}
Assessment: Acute viral pharyngitis confirmed.
Patient reports sore throat and cough for 3 days.

{
  "icd_predictions": [
    { "code": "J02.9", "confidence": 0.78 }
  ],
  "explanations": {
    "J02.9": [
      { "token": "pharyngitis", "importance": 0.42 }
    ]
  }
}
```
✔ Disease coding enabled
✔ Explainable prediction

⚠️ Ethical & Clinical Considerations

This system is NOT intended to:

Diagnose patients

Replace clinicians

Make final billing decisions

It is designed as a clinical decision support tool that:

Assists documentation workflows

Reduces manual effort

Enforces safety through context awareness

🚀 How to Run Locally
1️⃣ Start Backend
uvicorn app.main:app --reload --port 8001

2️⃣ Start Frontend
streamlit run frontend/app.py


Open:

http://localhost:8501

🏁 Final Note

This project prioritizes clinical correctness, safety, and interpretability over naive accuracy metrics, reflecting how real healthcare AI systems are built and evaluated.
