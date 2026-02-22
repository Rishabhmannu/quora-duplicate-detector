"""
Sentence Transformer embeddings for semantic similarity.
Uses MPS (Apple Silicon GPU) when available.
"""
import numpy as np

_embedding_model = None


def get_embedding_model(device: str = None):
    """Load Sentence Transformer model (cached singleton)."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        return _embedding_model
    except ImportError:
        return None


def embedding_cosine_similarity(q1: str, q2: str, model=None) -> float:
    """
    Compute cosine similarity between question embeddings.
    Returns 0.0 if model unavailable.
    """
    if model is None:
        model = get_embedding_model()
    if model is None:
        return 0.0

    embeddings = model.encode([q1, q2])
    a, b = embeddings[0], embeddings[1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
