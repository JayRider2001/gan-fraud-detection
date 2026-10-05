"""Shared raw-row → model-input conversion (log1p Amount, then scaler)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import META_PATH
from src.serving import load_bundle


def feature_names() -> list[str]:
    return json.loads(META_PATH.read_text())["feature_names"]


def log1p_amount(X: np.ndarray, names: list[str]) -> np.ndarray:
    """If Amount looks raw (non-negative), log1p it. Idempotent enough for Kaggle CSV."""
    out = np.array(X, dtype=np.float64, copy=True)
    if "Amount" not in names:
        return out
    i = names.index("Amount")
    if np.nanmin(out[:, i]) >= 0:
        out[:, i] = np.log1p(out[:, i])
    return out


def frame_to_raw(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = feature_names()
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df[names].to_numpy(dtype=np.float64), names


def transform(X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return scaled X, P(fraud), threshold. Amount may still be raw."""
    meta, scaler, model, thr = load_bundle()
    names = meta["feature_names"]
    if X_raw.shape[1] != len(names):
        raise ValueError(f"expected {len(names)} features, got {X_raw.shape[1]}")
    X = log1p_amount(X_raw, names)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1]
    return Xs, proba, thr
