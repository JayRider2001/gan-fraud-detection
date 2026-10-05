"""SHAP TreeExplainer on the shipped XGBoost. Top-tier analog: Noise / CSM BI / doc-fraud."""

from __future__ import annotations

import numpy as np

from src.config import SHAP_BG
from src.serving import load_bundle


def save_background(X_train: np.ndarray, n: int = 256, seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    n = min(n, len(X_train))
    idx = rng.choice(len(X_train), n, replace=False)
    np.save(SHAP_BG, X_train[idx].astype(np.float32))


_EXPLAINER = None


def _explainer(model, background: np.ndarray):
    import shap

    # Use the whole saved background. The default masker keeps only 100 rows.
    masker = shap.maskers.Independent(background, max_samples=len(background))
    return shap.TreeExplainer(model, data=masker, feature_perturbation="interventional")


def _get_explainer():
    """Build the TreeExplainer once. The background sample is the train-set draw."""
    global _EXPLAINER
    if _EXPLAINER is None:
        meta, _, model, _ = load_bundle()
        if not SHAP_BG.exists():
            raise FileNotFoundError("shap_background.npy missing; run detect or python -m src.explain")
        bg = np.load(SHAP_BG)
        _EXPLAINER = (meta["feature_names"], _explainer(model, bg))
    return _EXPLAINER


def _fraud_values(raw) -> np.ndarray:
    if isinstance(raw, list):
        values = np.asarray(raw[-1])
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[:, :, -1]
    return values


def shap_for_rows(Xs: np.ndarray, top_k: int = 8) -> list[dict]:
    """Xs is scaled. Returns per-row top contributing features toward fraud, plus the full vector."""
    names, exp = _get_explainer()
    values = _fraud_values(exp.shap_values(Xs))
    base = float(np.asarray(exp.expected_value).reshape(-1)[-1])
    out = []
    for i in range(len(Xs)):
        sv = np.asarray(values[i], dtype=np.float64).reshape(-1)
        order = np.argsort(-np.abs(sv))[:top_k]
        out.append(
            {
                "base_value": base,
                "vector": [float(v) for v in sv],
                "top": [
                    {
                        "feature": names[j],
                        "shap": float(sv[j]),
                        "value_scaled": float(Xs[i, j]),
                    }
                    for j in order
                ],
            }
        )
    return out


if __name__ == "__main__":
    from src.data import load_processed

    data = load_processed()
    save_background(data["X_train"])
    print(f"wrote {SHAP_BG}")
