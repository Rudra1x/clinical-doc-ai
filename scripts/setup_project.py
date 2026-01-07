import os

BASE_DIRS = [
    "services/ocr_service/app",
    "services/nlp_inference_service/app",
    "services/nlp_inference_service/models",
    "services/explainability_service/app",
    "frontend",
    "mlops/monitoring",
    "mlops/training",
    "mlops/pipelines",
    "infra",
    "data/raw",
    "data/processed",
    "data/samples",
    "scripts",
]

FILES = {
    "services/ocr_service/app/main.py": "",
    "services/ocr_service/app/ocr.py": "",
    "services/ocr_service/app/utils.py": "",
    "services/ocr_service/requirements.txt": "",
    "services/ocr_service/Dockerfile": "",

    "services/nlp_inference_service/app/main.py": "",
    "services/nlp_inference_service/app/model.py": "",
    "services/nlp_inference_service/app/inference.py": "",
    "services/nlp_inference_service/app/schemas.py": "",
    "services/nlp_inference_service/requirements.txt": "",
    "services/nlp_inference_service/Dockerfile": "",

    "services/explainability_service/app/main.py": "",
    "services/explainability_service/app/explain.py": "",
    "services/explainability_service/app/heatmaps.py": "",
    "services/explainability_service/requirements.txt": "",
    "services/explainability_service/Dockerfile": "",

    "frontend/streamlit_app.py": "",
    "frontend/requirements.txt": "",

    "mlops/monitoring/drift_detection.py": "",
    "mlops/monitoring/metrics.py": "",

    "mlops/training/train_biobert.py": "",
    "mlops/training/weak_supervision.py": "",
    "mlops/training/active_learning.py": "",

    "mlops/pipelines/retraining_pipeline.py": "",

    "infra/docker-compose.yml": "",
    "infra/nginx.conf": "",

    "README.md": "# ClinicalDocAI\n",
    ".gitignore": "__pycache__/\n.env\n*.pt\n*.onnx\n",
    "pyproject.toml": "",
}

def create_structure():
    for directory in BASE_DIRS:
        os.makedirs(directory, exist_ok=True)

    for file_path, content in FILES.items():
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(content)

    print("ClinicalDocAI project structure created successfully.")

if __name__ == "__main__":
    create_structure()