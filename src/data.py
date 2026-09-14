"""Download, split, and scale. Split happens BEFORE any oversampling."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_URL,
    META_PATH,
    PROCESSED,
    RAW_CSV,
    SCALER_PATH,
    SEED,
    TRAIN_SIZE,
    VAL_SIZE,
)


def download_csv(dest: Path = RAW_CSV) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[data] using cached {dest}")
        return dest
    print(f"[data] downloading {DATA_URL}")
    urlretrieve(DATA_URL, dest)
    print(f"[data] wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Keep V1–V28, Time, and log1p(Amount). Class is the label."""
    out = df.copy()
    out["Amount"] = np.log1p(out["Amount"].astype(np.float64))
    feature_names = [c for c in out.columns if c != "Class"]
    X = out[feature_names].to_numpy(dtype=np.float32)
    y = out["Class"].to_numpy(dtype=np.int32)
    return out, X, y, feature_names


def stratified_splits(X: np.ndarray, y: np.ndarray):
    """70 / 15 / 15, stratified. Test is never used for fitting anything."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=TRAIN_SIZE, stratify=y, random_state=SEED
    )
    relative_val = VAL_SIZE / (1.0 - TRAIN_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=relative_val, stratify=y_tmp, random_state=SEED
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare() -> dict:
    download_csv()
    df = pd.read_csv(RAW_CSV)
    n_fraud = int((df["Class"] == 1).sum())
    print(
        f"[data] {len(df):,} rows, {n_fraud} fraud "
        f"({100 * n_fraud / len(df):.3f}%)"
    )

    _, X, y, feature_names = engineer(df)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_splits(X, y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    def counts(split, labels):
        pos = int(labels.sum())
        print(f"[data] {split:5s}  n={len(labels):6d}  fraud={pos:4d}  rate={pos / len(labels):.4%}")

    counts("train", y_train)
    counts("val", y_val)
    counts("test", y_test)

    np.savez_compressed(
        PROCESSED,
        X_train=X_train_s,
        y_train=y_train,
        X_val=X_val_s,
        y_val=y_val,
        X_test=X_test_s,
        y_test=y_test,
        feature_names=np.array(feature_names),
    )
    joblib.dump(scaler, SCALER_PATH)
    meta = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_fraud": int(y_train.sum()),
        "val_fraud": int(y_val.sum()),
        "test_fraud": int(y_test.sum()),
        "amount_transform": "log1p then StandardScaler (scaler fit on train only)",
        "leakage_protocol": "split -> scale(train) -> GAN on train fraud only -> test once",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[data] saved {PROCESSED.name}, {SCALER_PATH.name}, {META_PATH.name}")
    return meta


def load_processed():
    if not PROCESSED.exists():
        raise FileNotFoundError("Run `python run.py data` first.")
    z = np.load(PROCESSED, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_scaler():
    return joblib.load(SCALER_PATH)


def fraud_only(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 1]


if __name__ == "__main__":
    prepare()
