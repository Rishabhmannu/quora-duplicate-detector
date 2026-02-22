"""
Fine-tune DistilBERT for Quora duplicate question detection.
Uses MPS on Apple Silicon when available.

Produces models/transformer/ (tokenizer + model)
Output saved to outputs/05_train_transformer_output.txt

Run from project root: python scripts/05_train_transformer.py
Quick test: python scripts/05_train_transformer.py --quick
"""
import sys
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Add project root for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
TRANSFORMER_DIR = MODELS_DIR / "transformer"
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"

sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess


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
    sample_size: int = 50000,
    random_state: int = 2,
    max_length: int = 128,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
):
    orig_stdout = sys.stdout
    tee = TeeOutput(orig_stdout)
    sys.stdout = tee

    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"ERROR: transformers/datasets not installed. Run: pip install transformers datasets accelerate")
        print(f"  {e}")
        sys.stdout = orig_stdout
        return

    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=["question1", "question2"])
        new_df = df.sample(min(sample_size, len(df)), random_state=random_state).copy()

        print("=" * 50)
        print("DISTILBERT FINE-TUNING: Duplicate Question Detection")
        print("=" * 50)
        print(f"Sample size: {len(new_df)}")

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model_name = "distilbert-base-uncased"
        print(f"\nLoading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Preprocess questions (consistent with rest of pipeline)
        def prepare_texts(row):
            q1 = preprocess(str(row["question1"])) or " "
            q2 = preprocess(str(row["question2"])) or " "
            return q1, q2

        print("\nTokenizing...")
        texts = [prepare_texts(row) for _, row in new_df.iterrows()]
        encodings = tokenizer(
            [t[0] for t in texts],
            [t[1] for t in texts],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="np",
        )

        labels = new_df["is_duplicate"].values

        # Stratified train/test split
        idx = np.arange(len(labels))
        idx_train, idx_eval = train_test_split(idx, test_size=0.2, random_state=1, stratify=labels)

        def slice_dict(d, indices):
            return {k: v[indices] for k, v in d.items()}

        enc_train = slice_dict(encodings, idx_train)
        enc_eval = slice_dict(encodings, idx_eval)
        labels_train = labels[idx_train].astype(np.int64)
        labels_eval = labels[idx_eval].astype(np.int64)

        train_dict = {
            "input_ids": enc_train["input_ids"],
            "attention_mask": enc_train["attention_mask"],
            "labels": labels_train,
        }
        eval_dict = {
            "input_ids": enc_eval["input_ids"],
            "attention_mask": enc_eval["attention_mask"],
            "labels": labels_eval,
        }
        if "token_type_ids" in encodings:
            train_dict["token_type_ids"] = enc_train["token_type_ids"]
            eval_dict["token_type_ids"] = enc_eval["token_type_ids"]

        train_dataset = Dataset.from_dict(train_dict)
        eval_dataset = Dataset.from_dict(eval_dict)

        training_args = TrainingArguments(
            output_dir=str(TRANSFORMER_DIR / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds),
                "precision": precision_score(labels, preds),
                "recall": recall_score(labels, preds),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        print("\nTraining...")
        trainer.train()

        print("\nFinal evaluation:")
        eval_result = trainer.evaluate()
        for k, v in eval_result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        # Save final model and tokenizer
        TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(TRANSFORMER_DIR))
        tokenizer.save_pretrained(str(TRANSFORMER_DIR))

        # Clean up checkpoints to save space
        import shutil
        ckpt_dir = TRANSFORMER_DIR / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

        print(f"\nSaved model and tokenizer to {TRANSFORMER_DIR}")

        # Quick inference test
        q1, q2 = (
            "What if we colonised Mars?",
            "What if Elon Musk decides to finally colonise Mars?",
        )
        q1_p, q2_p = preprocess(q1), preprocess(q2)
        inputs = tokenizer(q1_p, q2_p, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        proba = torch.softmax(logits, dim=-1)[0, 1].item()
        pred = 1 if proba >= 0.5 else 0
        print(f"\nTest: '{q1}' vs '{q2}'")
        print(f"  -> {'Duplicate' if pred else 'Not Duplicate'} (P={proba:.3f})")

        print("\n" + "=" * 50)
    finally:
        sys.stdout = orig_stdout

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / "05_train_transformer_output.txt"
    with open(out_path, "w") as f:
        f.write(tee.getvalue())
    print(f"\nOutput saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50000, help="Training samples (default 50k)")
    parser.add_argument("--quick", action="store_true", help="5K samples for quick test")
    parser.add_argument("--full", action="store_true", help="Use full 404K dataset")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    n = 5000 if args.quick else (404000 if args.full else args.sample)
    main(
        sample_size=n,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
