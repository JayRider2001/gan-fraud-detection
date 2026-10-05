"""Load the shipped detector. Generator is never loaded here."""

from __future__ import annotations

import json

from xgboost import XGBClassifier

from src.config import META_PATH, MODELS, THRESHOLD_PATH
from src.data import load_scaler

_BUNDLE = None


def load_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        meta = json.loads(META_PATH.read_text())
        thresholds = json.loads(THRESHOLD_PATH.read_text())
        scaler = load_scaler()
        model = XGBClassifier()
        model.load_model(MODELS["gan"])
        _BUNDLE = (meta, scaler, model, float(thresholds["gan"]))
    return _BUNDLE
