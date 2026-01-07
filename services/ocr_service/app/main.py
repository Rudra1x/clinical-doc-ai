from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="OCR Service",
    description="Extract text from medical documents",
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
    return {"status": "OCR service is healthy"}

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "text": "Dummy OCR output (Tesseract coming next)"
    }