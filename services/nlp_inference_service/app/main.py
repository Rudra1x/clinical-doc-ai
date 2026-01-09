from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import InferenceRequest, InferenceResponse, ICDPrediction
from .inference import run_inference

app = FastAPI(
    title="Clinical Document Intelligence API",
    description="Context-aware ICD-10 coding, explainable AI, and clinical NER",
    version="1.0.0",
)

# CORS (safe for demo / local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
def health_check():
    return {"status": "OK"}


# Main inference endpoint
@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    """
    Context-aware inference endpoint.
    """

    summary, icd_predictions, explanations, ner_entities = run_inference(
        text=request.document_text,
        explain=request.explain,
    )

    icd_objects = [
        ICDPrediction(**pred) for pred in icd_predictions
    ]

    return InferenceResponse(
        summary=summary,
        icd_predictions=icd_objects,
        explanations=explanations,
        ner_entities=ner_entities,
    )