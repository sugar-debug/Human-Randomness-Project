"""Generate the computer reference sequences.

One fair random (digits, flips) pair per participant, same lengths as the
human data. Seed 42 reproduces the main reference used in the project; the
project also re-checked results against 20 other computer references.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from features import FEATURE_COLS, parse_digits, parse_flips, sequence_features


def simulate(survey: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in survey.iterrows():
        d = rng.integers(0, 10, size=int(r["digit_len"])).tolist()
        b = rng.integers(0, 2, size=int(r["binary_len"])).tolist()
        rows.append(sequence_features(d, b))
    return pd.DataFrame(rows, columns=FEATURE_COLS)


def human_features(survey: pd.DataFrame) -> pd.DataFrame:
    rows = [
        sequence_features(parse_digits(r["digit_sequence"]), parse_flips(r["binary_sequence"]))
        for _, r in survey.iterrows()
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", default="data/survey_anonymized.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/processed/computer_features_seed42.csv")
    a = ap.parse_args()
    simulate(pd.read_csv(a.survey), a.seed).to_csv(a.out, index=False)
    print("wrote", a.out)
