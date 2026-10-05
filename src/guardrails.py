"""Schema and out-of-distribution checks. Fail closed on garbage rows, warn on drift."""

from __future__ import annotations

import numpy as np

from src.config import OOD_Z


def validate_finite(X: np.ndarray) -> None:
    if not np.isfinite(X).all():
        raise ValueError("non-finite values in features (NaN/inf)")


def ood_flags(Xs: np.ndarray, names: list[str], z: float = OOD_Z) -> list[dict]:
    """Xs is already scaled. |z| > OOD_Z means the row is far from train."""
    out = []
    absz = np.abs(Xs)
    for i in range(len(Xs)):
        hit = np.where(absz[i] > z)[0]
        out.append(
            {
                "is_ood": bool(len(hit) > 0),
                "n_ood_features": int(len(hit)),
                "ood_features": [names[j] for j in hit[:8]],
            }
        )
    return out
