"""Rebuild the labelled dataset for SMAP channel T-1.

Maps the anomaly intervals listed in data/labeled_anomalies.csv onto the raw
telemetry in data/raw_Channel_T-1.npy (interval ends are inclusive).

Output columns: 0..24 (the 25 input columns), Normal (1 = normal, -1 = anomaly),
Point Type (Normal / point / contextual).

Usage (from the repository root):
    python scripts/build_labels.py            # writes data/T-1_final_labeled.rebuilt.csv
    python scripts/build_labels.py --check    # also compares with data/T-1_final_labeled.csv
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

CHANNEL = "T-1"
DATA = Path(__file__).resolve().parent.parent / "data"


def build() -> pd.DataFrame:
    raw = np.load(DATA / "raw_Channel_T-1.npy")
    meta = pd.read_csv(DATA / "labeled_anomalies.csv")
    row = meta.loc[meta["chan_id"] == CHANNEL].iloc[0]

    sequences = ast.literal_eval(row["anomaly_sequences"])
    # classes are stored like "[point, contextual]" (not valid Python), so parse by hand
    classes = [c.strip() for c in row["class"].strip("[]").split(",")]
    assert len(sequences) == len(classes), "one class per anomaly sequence expected"

    df = pd.DataFrame(raw, columns=[str(i) for i in range(raw.shape[1])])
    df["Normal"] = 1.0
    df["Point Type"] = "Normal"
    for (start, end), kind in zip(sequences, classes):
        df.loc[start:end, "Normal"] = -1.0  # .loc slices include the end index
        df.loc[start:end, "Point Type"] = kind
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare with the existing labelled file")
    args = parser.parse_args()

    out = build()
    target = DATA / "T-1_final_labeled.rebuilt.csv"
    out.to_csv(target, index=False)
    print(f"wrote {target.name}: {len(out)} rows, {(out['Normal'] == -1).sum()} anomalous timesteps")
    print(out["Point Type"].value_counts().to_string())

    if args.check:
        ref = pd.read_csv(DATA / "T-1_final_labeled.csv")
        same_labels = (ref["Normal"].values == out["Normal"].values).all()
        same_types = (ref["Point Type"].values == out["Point Type"].values).all()
        same_values = np.allclose(ref.iloc[:, :25].values, out.iloc[:, :25].values)
        print(f"labels match: {same_labels} | point types match: {same_types} | values match: {same_values}")
