from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import InferenceRequest, InferenceResponse, ICDPrediction
from .inference import run_inference

app = FastAPI(
    title="NLP Inference Service",
    description="Clinical NLP for ICD coding and summarization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "NLP inference service is healthy"}

@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    summary, icd_preds, entities = run_inference(request.document_text)

    icd_objects = [
        ICDPrediction(**pred) for pred in icd_preds
    ]

    return InferenceResponse(
        summary=summary,
        icd_predictions=icd_objects,
        entities=entities
    )