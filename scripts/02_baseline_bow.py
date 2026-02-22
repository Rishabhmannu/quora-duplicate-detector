"""
Baseline: BoW only, no handcrafted features.
Output saved to outputs/02_baseline_bow_output.txt
Run from project root: python scripts/02_baseline_bow.py
"""
import sys
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class TeeOutput:
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


def main(sample_size: int = 30000, random_state: int = 1):
    orig_stdout = sys.stdout
    tee = TeeOutput(orig_stdout)
    sys.stdout = tee

    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=["question1", "question2"])
        new_df = df.sample(sample_size, random_state=random_state)

        print("=" * 50)
        print("BASELINE: BoW only")
        print("=" * 50)
        print(f"Sample size: {len(new_df)}")

        ques_df = new_df[["question1", "question2"]]
        questions = list(ques_df["question1"]) + list(ques_df["question2"])

        cv = CountVectorizer(max_features=3000)
        q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)

        temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
        temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)
        temp_df = pd.concat([temp_df1, temp_df2], axis=1)
        temp_df["is_duplicate"] = new_df["is_duplicate"]

        X = temp_df.iloc[:, :-1].values
        y = temp_df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        print("\n--- Random Forest ---")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        print("\n--- XGBoost ---")
        xgb = XGBClassifier(eval_metric="logloss")
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        print("\n" + "=" * 50)
    finally:
        sys.stdout = orig_stdout

    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "02_baseline_bow_output.txt", "w") as f:
        f.write(tee.getvalue())
    print(f"Output saved to {OUTPUTS_DIR / '02_baseline_bow_output.txt'}")
    return rf, xgb, cv


if __name__ == "__main__":
    main()
