"""
Helper module for Streamlit app.
Loads model artifacts and delegates to src for feature extraction.
Supports classical (RF/XGBoost) and transformer (DistilBERT) models.
"""
import pickle
import json
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path for src imports
_project_root = Path(__file__).resolve().parent.parent
import sys

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.feature_engineering import query_point_creator as _query_point_creator
from src.embeddings import get_embedding_model

# Paths
_models_dir = _project_root / "models"
_app_dir = Path(__file__).resolve().parent
_transformer_dir = _models_dir / "transformer"
_inference_times_path = _models_dir / "inference_times.json"


def _ensure_models_from_hf():
    """Download models from HF Hub if not present and HF_MODEL_REPO is set."""
    import os
    repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id or (_models_dir / "model.pkl").exists():
        return
    try:
        from huggingface_hub import snapshot_download
        _models_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=str(_models_dir))
    except Exception as e:
        print(f"HF Hub download skipped or failed: {e}")


# Try HF Hub download when models missing (for HF Spaces deployment)
_ensure_models_from_hf()

# Classical model artifacts (lazy loaded)
_classical_model = None
_classical_cv = None
_embedding_model = None

# Transformer (lazy loaded)
_transformer_model = None
_transformer_tokenizer = None


def _get_cv_path():
    return _models_dir / "cv.pkl" if (_models_dir / "cv.pkl").exists() else _app_dir / "cv.pkl"


def _get_model_path():
    return _models_dir / "model.pkl" if (_models_dir / "model.pkl").exists() else _app_dir / "model.pkl"


def get_available_models() -> list:
    """Return list of available model identifiers."""
    available = []
    if _get_model_path().exists() and _get_cv_path().exists():
        available.append("classical")
    if (_transformer_dir / "config.json").exists():
        available.append("transformer")
    return available


def get_inference_times() -> dict:
    """Load benchmark results from models/inference_times.json."""
    if not _inference_times_path.exists():
        return {}
    try:
        with open(_inference_times_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_classical():
    global _classical_model, _classical_cv, _embedding_model
    if _classical_model is None:
        _classical_model = pickle.load(open(_get_model_path(), "rb"))
        _classical_cv = pickle.load(open(_get_cv_path(), "rb"))
        _embedding_model = get_embedding_model()
    return _classical_model, _classical_cv, _embedding_model


def _load_transformer():
    global _transformer_model, _transformer_tokenizer
    if _transformer_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        _transformer_tokenizer = AutoTokenizer.from_pretrained(str(_transformer_dir))
        _transformer_model = AutoModelForSequenceClassification.from_pretrained(str(_transformer_dir))
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _transformer_model = _transformer_model.to(device)
        _transformer_model.eval()
    return _transformer_model, _transformer_tokenizer


def query_point_creator(q1: str, q2: str):
    """Build feature vector for classical model. Uses shared src modules + embeddings."""
    _, cv, emb = _load_classical()
    return _query_point_creator(q1, q2, cv, embedding_model=emb)


def predict_classical(q1: str, q2: str) -> Tuple[int, float]:
    """Predict using classical model. Returns (pred, proba)."""
    model, cv, emb = _load_classical()
    feat = _query_point_creator(q1, q2, cv, embedding_model=emb)
    proba = model.predict_proba(feat)[0, 1]
    pred = int(proba >= 0.5)
    return pred, float(proba)


def predict_transformer(q1: str, q2: str) -> Tuple[int, float]:
    """Predict using DistilBERT. Returns (pred, proba)."""
    from src.preprocessing import preprocess
    import torch

    model, tokenizer = _load_transformer()
    q1_p, q2_p = preprocess(q1), preprocess(q2)
    inputs = tokenizer(
        q1_p, q2_p,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    proba = torch.softmax(logits, dim=-1)[0, 1].item()
    pred = 1 if proba >= 0.5 else 0
    return pred, float(proba)


def predict(q1: str, q2: str, model_type: str) -> Tuple[int, float]:
    """Unified prediction. model_type: 'classical' or 'transformer'."""
    if model_type == "classical":
        return predict_classical(q1, q2)
    if model_type == "transformer":
        return predict_transformer(q1, q2)
    raise ValueError(f"Unknown model_type: {model_type}")


def get_model_display_name(model_type: str) -> str:
    """Human-readable name for model selector."""
    return {"classical": "Classical (RF/XGBoost + TF-IDF)", "transformer": "DistilBERT (Transformer)"}.get(
        model_type, model_type
    )
