"""
Full pipeline: preprocessing + 25 features (incl. embedding cosine) + TF-IDF -> train -> save.
Produces model.pkl, cv.pkl in models/

Improvements (per project-plan.md):
- Sentence Transformer embeddings (MPS on Apple Silicon)
- TF-IDF, Stratified 5-Fold CV, full metrics
- Output saved to outputs/04_train_and_save_output.txt
- tqdm for progress bars

Run from project root: python scripts/04_train_and_save.py
"""
import sys
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# Add project root for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"

sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import query_point_creator
from src.model import evaluate_model, print_metrics, stratified_cv_evaluate
from src.embeddings import get_embedding_model


class TeeOutput:
    """Print to console and capture to string."""

    def __init__(self, real_stdout):
        self.real_stdout = real_stdout
        self.buffer = StringIO()

    def write(self, s):
        self.real_stdout.write(s)
        self.buffer.write(s)

    def flush(self):
        self.real_stdout.flush()
        self.buffer.flush()

    def getvalue(self):
        return self.buffer.getvalue()

    def isatty(self):
        return getattr(self.real_stdout, "isatty", lambda: False)()


def main(
    sample_size: int = 30000,
    random_state: int = 2,
    use_tfidf: bool = True,
    use_embeddings: bool = True,
    n_folds: int = 5,
):
    orig_stdout = sys.stdout
    tee = TeeOutput(orig_stdout)
    sys.stdout = tee

    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=["question1", "question2"])
        new_df = df.sample(sample_size, random_state=random_state).copy()

        vec_name = "TF-IDF" if use_tfidf else "BoW"
        emb_str = "+ embeddings" if use_embeddings else ""
        print("=" * 50)
        print(f"FULL PIPELINE: Preprocess + 25 features {emb_str} + {vec_name}")
        print("=" * 50)
        print(f"Sample size: {len(new_df)}")

        # Load embedding model
        embedding_model = None
        if use_embeddings:
            print("\nLoading Sentence Transformer (all-MiniLM-L6-v2)...")
            embedding_model = get_embedding_model()
            if embedding_model is None:
                print("WARNING: sentence-transformers not available, skipping embeddings")
            else:
                print("Embedding model loaded (MPS/CPU)")

        # Vectorizer
        Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
        vectorizer = Vectorizer(max_features=3000)
        questions = list(new_df["question1"]) + list(new_df["question2"])
        vectorizer.fit(questions)

        # Build feature matrix
        print("\nBuilding features...")
        features_list = []
        for _, row in tqdm(
            new_df.iterrows(),
            total=len(new_df),
            desc="Feature extraction",
            unit="rows",
        ):
            feat = query_point_creator(
                row["question1"], row["question2"], vectorizer, embedding_model
            )
            features_list.append(feat)

        X = np.vstack(features_list)
        y = new_df["is_duplicate"].values

        print(f"\nFeature matrix shape: {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1, stratify=y
        )

        # Stratified K-Fold CV
        print(f"\n--- Stratified {n_folds}-Fold Cross-Validation ---")
        print("Random Forest:")
        rf = RandomForestClassifier()
        mean_rf, _ = stratified_cv_evaluate(
            rf, X_train, y_train, n_folds=n_folds
        )
        print_metrics(mean_rf, "RF (CV mean)")

        print("XGBoost:")
        xgb = XGBClassifier(eval_metric="logloss")
        mean_xgb, _ = stratified_cv_evaluate(
            xgb, X_train, y_train, n_folds=n_folds
        )
        print_metrics(mean_xgb, "XGBoost (CV mean)")

        # Train final models
        print("--- Final models (trained on 80% train) ---")
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        print("\nRandom Forest (test set):")
        rf_metrics = evaluate_model(rf, X_test, y_test)
        print_metrics(rf_metrics)
        print("Confusion matrix:")
        print(confusion_matrix(y_test, rf.predict(X_test)))

        print("\nXGBoost (test set):")
        xgb_metrics = evaluate_model(xgb, X_test, y_test)
        print_metrics(xgb_metrics)
        print("Confusion matrix:")
        print(confusion_matrix(y_test, xgb.predict(X_test)))

        use_rf = rf_metrics["f1"] >= xgb_metrics["f1"]
        best_model = rf if use_rf else xgb
        best_name = "RandomForest" if use_rf else "XGBoost"

        # Save artifacts
        MODELS_DIR.mkdir(exist_ok=True)
        import pickle

        with open(MODELS_DIR / "model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        with open(MODELS_DIR / "cv.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        print(f"\nSaved {best_name} model.pkl and cv.pkl to {MODELS_DIR}")

        # Inference test
        q1, q2 = (
            "What if we colonised Mars?",
            "What if Elon Musk decides to finally colonise Mars?",
        )
        feat = query_point_creator(q1, q2, vectorizer, embedding_model)
        pred = best_model.predict(feat)[0]
        proba = best_model.predict_proba(feat)[0, 1]
        print(f"\nTest: '{q1}' vs '{q2}'")
        print(f"  -> {'Duplicate' if pred else 'Not Duplicate'} (P={proba:.3f})")

        print("\n" + "=" * 50)
    finally:
        sys.stdout = orig_stdout

    # Save output to file
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / "04_train_and_save_output.txt"
    with open(out_path, "w") as f:
        f.write(tee.getvalue())
    print(f"\nOutput saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=30000)
    parser.add_argument("--quick", action="store_true", help="5K samples")
    parser.add_argument("--bow", action="store_true", help="Use BoW not TF-IDF")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip Sentence Transformers")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    n = 5000 if args.quick else args.sample
    main(
        sample_size=n,
        use_tfidf=not args.bow,
        use_embeddings=not args.no_embeddings,
        n_folds=args.folds,
    )
