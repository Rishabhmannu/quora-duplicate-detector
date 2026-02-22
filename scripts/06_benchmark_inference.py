"""
Benchmark inference time for all models: RF/XGBoost (TF-IDF+features) and DistilBERT.
Saves results to models/inference_times.json.

Run from project root: python scripts/06_benchmark_inference.py
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TRANSFORMER_DIR = MODELS_DIR / "transformer"

sys.path.insert(0, str(PROJECT_ROOT))


def benchmark_classical(n_warmup=5, n_runs=50):
    """Benchmark RF/XGBoost model (single prediction)."""
    import pickle
    from src.feature_engineering import query_point_creator
    from src.embeddings import get_embedding_model

    model_path = MODELS_DIR / "model.pkl"
    cv_path = MODELS_DIR / "cv.pkl"
    if not model_path.exists() or not cv_path.exists():
        return None

    model = pickle.load(open(model_path, "rb"))
    cv = pickle.load(open(cv_path, "rb"))
    emb = get_embedding_model()

    q1, q2 = "What is the capital of India?", "Which city is India's capital?"

    def get_feat():
        feat = query_point_creator(q1, q2, cv, emb)
        n_expected = getattr(model, "n_features_in_", feat.shape[1])
        if feat.shape[1] < n_expected:
            pad = np.zeros((1, n_expected - feat.shape[1]))
            feat = np.hstack([feat, pad])
        return feat

    # Warmup
    for _ in range(n_warmup):
        _ = model.predict_proba(get_feat())

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict_proba(get_feat())
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
    }


def benchmark_transformer(n_warmup=5, n_runs=50):
    """Benchmark DistilBERT model (single prediction)."""
    if not (TRANSFORMER_DIR / "config.json").exists():
        return None

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from src.preprocessing import preprocess

    tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_DIR))
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    q1, q2 = "What is the capital of India?", "Which city is India's capital?"
    q1_p, q2_p = preprocess(q1), preprocess(q2)
    max_length = 128

    # Warmup
    for _ in range(n_warmup):
        inputs = tokenizer(q1_p, q2_p, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs).logits

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        inputs = tokenizer(q1_p, q2_p, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs).logits
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
    }


def main():
    print("=" * 50)
    print("INFERENCE BENCHMARK")
    print("=" * 50)

    results = {}

    print("\n1. Classical (RF/XGBoost + TF-IDF + features)...")
    classical = benchmark_classical()
    if classical:
        results["classical"] = classical
        print(f"   mean: {classical['mean_ms']:.2f} ms, median: {classical['median_ms']:.2f} ms")
    else:
        print("   SKIP: model.pkl or cv.pkl not found")

    print("\n2. DistilBERT (transformer)...")
    transformer = benchmark_transformer()
    if transformer:
        results["transformer"] = transformer
        print(f"   mean: {transformer['mean_ms']:.2f} ms, median: {transformer['median_ms']:.2f} ms")
    else:
        print("   SKIP: transformer model not found (run 05_train_transformer.py first)")

    out_path = MODELS_DIR / "inference_times.json"
    MODELS_DIR.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
