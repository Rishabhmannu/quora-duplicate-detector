"""
EDA for Quora Question Pairs dataset.
Output saved to outputs/01_eda_output.txt
Run from project root: python scripts/01_eda.py
"""
import sys
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


def main(show_plots: bool = False):
    orig_stdout = sys.stdout
    tee = TeeOutput(orig_stdout)
    sys.stdout = tee

    try:
        df = pd.read_csv(DATA_PATH)

        print("=" * 50)
        print("QUORA QUESTION PAIRS - EDA")
        print("=" * 50)
        print(f"\nShape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")

        print("\n--- Sample ---")
        print(df.sample(5).to_string())

        print("\n--- Info ---")
        df.info()

        print("\n--- Missing values ---")
        print(df.isnull().sum())

        print("\n--- Duplicate rows ---")
        print(f"Count: {df.duplicated().sum()}")

        print("\n--- Target distribution ---")
        counts = df["is_duplicate"].value_counts()
        pcts = counts / df["is_duplicate"].count() * 100
        for idx, (val, cnt) in enumerate(counts.items()):
            print(f"  is_duplicate={val}: {cnt} ({pcts.iloc[idx]:.2f}%)")

        qid = pd.Series(df["qid1"].tolist() + df["qid2"].tolist())
        print(f"\n--- Unique questions: {np.unique(qid).shape[0]}")
        repeated = (qid.value_counts() > 1).sum()
        print(f"--- Questions repeated: {repeated}")

        if show_plots:
            OUTPUTS_DIR.mkdir(exist_ok=True)
            df["is_duplicate"].value_counts().plot(kind="bar")
            plt.title("Duplicate vs Non-duplicate")
            plt.savefig(OUTPUTS_DIR / "eda_target_dist.png", dpi=100)
            plt.close()
            print("\nSaved plot to outputs/eda_target_dist.png")

        print("\n" + "=" * 50)
    finally:
        sys.stdout = orig_stdout

    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "01_eda_output.txt", "w") as f:
        f.write(tee.getvalue())
    print(f"Output saved to {OUTPUTS_DIR / '01_eda_output.txt'}")
    return df


if __name__ == "__main__":
    show_plots = "--plots" in sys.argv
    main(show_plots=show_plots)
