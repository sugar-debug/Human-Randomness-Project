"""Train the logistic regression behind the web demo and print its parameters.

Humans (label 1) vs simulated computer sequences (label 0), 12 features,
standardised, logistic regression. Prints repeated cross-validated accuracy and
writes the weights as JSON in the same shape as the constant `M` in
app/index.html.

Note: the published demo was trained with a larger pool of simulated computer
sequences for the 95% ranges, so numbers here will be close to, not identical
to, the ones baked into app/index.html.

Usage:  python src/train_export.py --out results/model_params.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS
from simulate_computer import human_features, simulate


def main(survey_path: str, seed: int, n_comp_ref: int, out: str) -> None:
    survey = pd.read_csv(survey_path)
    hum = human_features(survey)
    comp = simulate(survey, seed)

    X = pd.concat([hum, comp], ignore_index=True)[FEATURE_COLS].to_numpy()
    y = np.r_[np.ones(len(hum)), np.zeros(len(comp))]

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()
    pipe.fit(X, y)
    sc, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]

    # Reference ranges: 95% interval across many simulated computer participants
    ref = pd.concat([simulate(survey, seed + 1 + i) for i in range(n_comp_ref)])
    params = {
        "cols": FEATURE_COLS,
        "mean": sc.mean_.tolist(),
        "scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "cv_acc": float(acc),
        "comp_lo": ref[FEATURE_COLS].quantile(0.025).tolist(),
        "comp_hi": ref[FEATURE_COLS].quantile(0.975).tolist(),
        "comp_med": ref[FEATURE_COLS].median().tolist(),
        "hum_med": hum[FEATURE_COLS].median().tolist(),
    }
    with open(out, "w") as f:
        json.dump(params, f, indent=2)
    print(f"cross-validated accuracy: {acc:.3f}  ->  {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", default="data/survey_anonymized.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-comp-ref", type=int, default=200)
    ap.add_argument("--out", default="results/model_params.json")
    a = ap.parse_args()
    main(a.survey, a.seed, a.n_comp_ref, a.out)
