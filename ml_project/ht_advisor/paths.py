from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_PROJECT_DIR = PROJECT_ROOT / "ml_project"
EXTRACTED_DATA_DIR = ML_PROJECT_DIR / "extracted_data"
CORPUS_PDF_DIR = ML_PROJECT_DIR / "corpus_pdfs"
CURATED_DATA_DIR = ML_PROJECT_DIR / "curated_data"
MODEL_OUTPUT_DIR = ML_PROJECT_DIR / "model_outputs"
DASHBOARD_DIR = ML_PROJECT_DIR / "dashboard"

