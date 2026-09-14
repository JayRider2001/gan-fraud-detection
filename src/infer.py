"""Score new rows with the GAN-augmented XGBoost. The generator is NOT loaded."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import META_PATH, MODELS, THRESHOLD_PATH
from src.data import load_scaler


def load_bundle():
    meta = json.loads(META_PATH.read_text())
    thresholds = json.loads(THRESHOLD_PATH.read_text())
    scaler = load_scaler()
    model = XGBClassifier()
    model.load_model(MODELS["gan"])
    return meta, scaler, model, float(thresholds["gan"])


def score_matrix(X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X_raw must already have Amount = log1p(original Amount), columns in training order."""
    meta, scaler, model, thr = load_bundle()
    if X_raw.shape[1] != meta["n_features"]:
        raise ValueError(f"expected {meta['n_features']} features, got {X_raw.shape[1]}")
    Xs = scaler.transform(X_raw)
    proba = model.predict_proba(Xs)[:, 1]
    flag = proba >= thr
    return proba, flag


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Score transactions with the GAN-augmented detector.")
    p.add_argument("--csv", type=str, help="CSV with the same columns as the training features (Time, V1..V28, Amount). Amount is the RAW amount; log1p is applied here.")
    p.add_argument("--values", type=str, help="Comma-separated raw feature vector in training order, Amount still raw.")
    args = p.parse_args(argv)

    meta, scaler, model, thr = load_bundle()
    names = meta["feature_names"]
    print(f"features ({len(names)}): {', '.join(names)}")
    print(f"threshold (val-chosen, gan model): {thr:.4f}")
    print("note: generator is not used at inference.\n")

    if args.csv:
        df = pd.read_csv(args.csv)
        missing = [c for c in names if c not in df.columns]
        if missing:
            print(f"missing columns: {missing}", file=sys.stderr)
            return 2
        X = df[names].to_numpy(dtype=np.float64)
        # Amount in the saved feature list is the log1p'd one; the CSV from Kaggle has raw Amount.
        if "Amount" in names:
            amt_idx = names.index("Amount")
            # If values look like raw amounts (non-negative, some > 10), log1p them.
            if np.nanmin(X[:, amt_idx]) >= 0:
                X[:, amt_idx] = np.log1p(X[:, amt_idx])
        proba, flag = score_matrix(X)
        out = df.copy()
        out["p_fraud"] = proba
        out["flag"] = flag.astype(int)
        print(out[["p_fraud", "flag"]].head(20).to_string(index=False))
        print(f"\nflagged {int(flag.sum())} / {len(flag)}")
        return 0

    if args.values:
        raw = np.fromstring(args.values, sep=",", dtype=np.float64).reshape(1, -1)
        X = raw.copy()
        names = json.loads(META_PATH.read_text())["feature_names"]
        if "Amount" in names:
            amt_idx = names.index("Amount")
            if X[0, amt_idx] >= 0:
                X[0, amt_idx] = np.log1p(X[0, amt_idx])
        proba, flag = score_matrix(X)
        print(f"P(fraud) = {proba[0]:.4f}   flag = {int(flag[0])}   (thr={thr:.4f})")
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
