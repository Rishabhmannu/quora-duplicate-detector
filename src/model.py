"""
Model training and evaluation utilities.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, prefix: str = ""):
    """
    Compute full evaluation metrics for a binary classifier.
    Returns dict of metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["log_loss"] = log_loss(y_test, y_proba)
        except ValueError:
            metrics["log_loss"] = float("nan")
        try:
            metrics["auc_roc"] = roc_auc_score(y_test, y_proba)
        except ValueError:
            metrics["auc_roc"] = float("nan")

    return metrics


def print_metrics(metrics: dict, prefix: str = ""):
    """Print metrics in a readable format."""
    p = f"{prefix} " if prefix else ""
    print(f"\n--- {p}Metrics ---")
    for name, val in metrics.items():
        if isinstance(val, float) and not np.isnan(val):
            print(f"  {name}: {val:.4f}")
        else:
            print(f"  {name}: {val}")
    print()


def stratified_cv_evaluate(model, X, y, n_folds: int = 5, random_state: int = 42):
    """
    Run Stratified K-Fold CV and return mean metrics.
    """
    from tqdm import tqdm

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_metrics = []
    for fold, (train_idx, val_idx) in tqdm(
        enumerate(skf.split(X, y)),
        total=n_folds,
        desc="CV folds",
        unit="fold",
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        m = evaluate_model(model_clone, X_val, y_val)
        fold_metrics.append(m)
        print(f"  Fold {fold + 1}: F1={m['f1']:.4f}, AUC={m.get('auc_roc', 0):.4f}")

    # Mean across folds
    mean_metrics = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics if not (isinstance(m[key], float) and np.isnan(m[key]))]
        mean_metrics[key] = np.mean(vals) if vals else float("nan")

    return mean_metrics, fold_metrics
