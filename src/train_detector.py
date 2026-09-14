"""Three detectors, identical XGBoost, different training rows.

weighted : raw train + scale_pos_weight
smote    : SMOTE to the same minority count the GAN will reach
gan      : real train + WGAN-GP fakes

Thresholds are chosen on val. Test is scored once.
"""

from __future__ import annotations

import json

import numpy as np
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.config import METRICS_PATH, MODELS, SEED, SYNTH_PATH, THRESHOLD_PATH, XGB_PARAMS
from src.data import fraud_only, load_processed
from src.evaluate import best_f1_threshold, full_report
from src.generate import n_synth_for
from src.plots import plot_confusion, plot_pr_curves, plot_tsne


def _xgb(scale_pos_weight: float | None = None) -> XGBClassifier:
    params = dict(XGB_PARAMS)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return XGBClassifier(**params)


def _fit_and_score(name: str, model: XGBClassifier, X_tr, y_tr, X_val, y_val, X_test, y_test) -> dict:
    print(f"[detect] fitting {name}  train={len(y_tr):,}  fraud={int(y_tr.sum())}")
    model.fit(X_tr, y_tr)
    model.save_model(MODELS[name])

    val_p = model.predict_proba(X_val)[:, 1]
    test_p = model.predict_proba(X_test)[:, 1]
    thr, val_f1 = best_f1_threshold(y_val, val_p)
    print(f"[detect] {name}: val-chosen threshold={thr:.4f}  val F1={val_f1:.3f}")

    report = {
        "val_f1_at_choice": float(val_f1),
        "val": full_report(y_val, val_p, thr),
        "test": full_report(y_test, test_p, thr),
    }
    plot_confusion(
        y_test,
        test_p,
        thr,
        title=f"{name}  (test, thr={thr:.3f})",
        filename=f"cm_{name}.png",
    )
    return report, test_p, thr


def train() -> dict:
    data = load_processed()
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_te, y_te = data["X_test"], data["y_test"]

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    target_pos = n_pos + n_synth_for(n_pos)
    print(f"[detect] train fraud={n_pos}  target minority after oversample={target_pos}")

    results = {}
    test_probas = {}
    thresholds = {}

    # 1. class-weighted, no extra rows
    w = n_neg / max(n_pos, 1)
    model_w = _xgb(scale_pos_weight=w)
    results["weighted"], test_probas["weighted"], thresholds["weighted"] = _fit_and_score(
        "weighted", model_w, X_tr, y_tr, X_val, y_val, X_te, y_te
    )
    results["weighted"]["scale_pos_weight"] = w
    results["weighted"]["n_train"] = int(len(y_tr))
    results["weighted"]["n_train_fraud"] = n_pos

    # 2. SMOTE to the same minority count
    k = max(1, min(5, n_pos - 1))
    smote = SMOTE(sampling_strategy={1: target_pos}, k_neighbors=k, random_state=SEED)
    X_sm, y_sm = smote.fit_resample(X_tr, y_tr)
    model_s = _xgb()
    results["smote"], test_probas["smote"], thresholds["smote"] = _fit_and_score(
        "smote", model_s, X_sm, y_sm, X_val, y_val, X_te, y_te
    )
    results["smote"]["n_train"] = int(len(y_sm))
    results["smote"]["n_train_fraud"] = int(y_sm.sum())

    # 3. GAN-augmented
    if not SYNTH_PATH.exists():
        raise FileNotFoundError("Run `python run.py gan` first (needs synthetic_fraud.npy).")
    fake = np.load(SYNTH_PATH)
    # Match SMOTE's minority count: keep all real fraud + enough fakes.
    n_needed = target_pos - n_pos
    if len(fake) < n_needed:
        raise RuntimeError(f"Need {n_needed} fakes, have {len(fake)}. Re-run generate.")
    fake = fake[:n_needed]
    X_gan = np.vstack([X_tr, fake])
    y_gan = np.concatenate([y_tr, np.ones(len(fake), dtype=y_tr.dtype)])
    model_g = _xgb()
    results["gan"], test_probas["gan"], thresholds["gan"] = _fit_and_score(
        "gan", model_g, X_gan, y_gan, X_val, y_val, X_te, y_te
    )
    results["gan"]["n_train"] = int(len(y_gan))
    results["gan"]["n_train_fraud"] = int(y_gan.sum())
    results["gan"]["n_synthetic"] = int(len(fake))

    plot_pr_curves({k: (y_te, test_probas[k]) for k in test_probas})
    plot_tsne(fraud_only(X_tr, y_tr), fake, X_tr[y_tr == 0])

    METRICS_PATH.write_text(json.dumps(results, indent=2))
    THRESHOLD_PATH.write_text(json.dumps(thresholds, indent=2))
    _print_table(results)
    return results


def _print_table(results: dict) -> None:
    print("\n=== TEST SET (frozen, threshold chosen on val) ===")
    hdr = f"{'model':<12} {'AUPRC':>8} {'ROC':>8} {'P':>8} {'R':>8} {'F1':>8} {'FP':>6} {'FN':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name in ("weighted", "smote", "gan"):
        t = results[name]["test"]
        print(
            f"{name:<12} {t['auprc']:8.4f} {t['roc_auc']:8.4f} "
            f"{t['precision']:8.4f} {t['recall']:8.4f} {t['f1']:8.4f} "
            f"{t['fp']:6d} {t['fn']:6d}"
        )
    print(f"\n[detect] wrote {METRICS_PATH}")


if __name__ == "__main__":
    train()
