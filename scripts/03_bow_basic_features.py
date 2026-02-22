"""
BoW + 7 basic handcrafted features.
Output saved to outputs/03_bow_basic_features_output.txt
Run from project root: python scripts/03_bow_basic_features.py
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


def common_words(row):
    w1 = set(word.lower().strip() for word in row["question1"].split())
    w2 = set(word.lower().strip() for word in row["question2"].split())
    return len(w1 & w2)


def total_words(row):
    w1 = set(word.lower().strip() for word in row["question1"].split())
    w2 = set(word.lower().strip() for word in row["question2"].split())
    return len(w1) + len(w2)


def main(sample_size: int = 30000, random_state: int = 2):
    orig_stdout = sys.stdout
    tee = TeeOutput(orig_stdout)
    sys.stdout = tee

    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=["question1", "question2"])
        new_df = df.sample(sample_size, random_state=random_state)

        print("=" * 50)
        print("BoW + 7 basic features")
        print("=" * 50)
        print(f"Sample size: {len(new_df)}")

        new_df["q1_len"] = new_df["question1"].str.len()
        new_df["q2_len"] = new_df["question2"].str.len()
        new_df["q1_num_words"] = new_df["question1"].apply(lambda r: len(r.split()))
        new_df["q2_num_words"] = new_df["question2"].apply(lambda r: len(r.split()))
        new_df["word_common"] = new_df.apply(common_words, axis=1)
        new_df["word_total"] = new_df.apply(total_words, axis=1)
        new_df["word_share"] = round(new_df["word_common"] / new_df["word_total"], 2)

        ques_df = new_df[["question1", "question2"]]
        questions = list(ques_df["question1"]) + list(ques_df["question2"])
        cv = CountVectorizer(max_features=3000)
        q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)
        temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
        temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)
        temp_df = pd.concat([temp_df1, temp_df2], axis=1)

        final_df = new_df.drop(columns=["id", "qid1", "qid2", "question1", "question2"])
        final_df = pd.concat([final_df, temp_df], axis=1)

        X = final_df.iloc[:, 1:].values
        y = final_df.iloc[:, 0].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        print("\n--- Random Forest ---")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"Accuracy: {acc:.4f}")

        print("\n--- XGBoost ---")
        xgb = XGBClassifier(eval_metric="logloss")
        xgb.fit(X_train, y_train)
        acc = accuracy_score(y_test, xgb.predict(X_test))
        print(f"Accuracy: {acc:.4f}")

        print("\n" + "=" * 50)
    finally:
        sys.stdout = orig_stdout

    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "03_bow_basic_features_output.txt", "w") as f:
        f.write(tee.getvalue())
    print(f"Output saved to {OUTPUTS_DIR / '03_bow_basic_features_output.txt'}")
    return rf, xgb, cv


if __name__ == "__main__":
    main()
