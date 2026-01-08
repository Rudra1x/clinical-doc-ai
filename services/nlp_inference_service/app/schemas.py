from pydantic import BaseModel
from typing import List, Dict


class InferenceRequest(BaseModel):
    document_text: str
    explain:bool = False


class ICDPrediction(BaseModel):
    code: str
    description: str
    confidence: float


class TokenImportance(BaseModel):
    token: str
    importance: float


class InferenceResponse(BaseModel):
    summary: str
    icd_predictions: List[ICDPrediction]
    entities: Dict[str, List[TokenImportance]]