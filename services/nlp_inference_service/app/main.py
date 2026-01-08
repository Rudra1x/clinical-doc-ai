from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import InferenceRequest, InferenceResponse, ICDPrediction
from .inference import run_inference
from .ner import run_ner

app = FastAPI(
    title="Clinical NLP Inference Service",
    description="ICD-10 prediction, explainability, and clinical NER",
    version="1.0.0",
)

# CORS (safe for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
def health_check():
    return {"status": "NLP inference service is healthy"}


# Main inference endpoint
@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    """
    Run ICD prediction with OPTIONAL explainability and NER.
    """

    # ICD prediction (+ optional explainability)
    summary, icd_predictions, explanations = run_inference(
        text=request.document_text,
        explain=getattr(request, "explain", False),
    )

    # Convert ICD predictions to schema objects
    icd_objects = [
        ICDPrediction(**pred) for pred in icd_predictions
    ]

    # Named Entity Recognition
    ner_entities = run_ner(request.document_text)

    # Merge explainability + NER into a single entities dict
    return InferenceResponse(
        summary=summary,
        icd_predictions=icd_objects,
        explanations=explanations,
        ner_entities=ner_entities,
    )