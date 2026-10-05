"""Production scoring API. Generator is not loaded."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.audit import log_score, recent, review
from src.case_agent import run_case
from src.config import METRICS_PATH, XAI_MODEL
from src.explain import shap_for_rows
from src.features import feature_names, transform
from src.guardrails import ood_flags, validate_finite

app = FastAPI(
    title="GAN fraud detector",
    description="XGBoost scorer trained with WGAN-GP synthetic fraud, plus a tool-calling case agent. No generator at inference.",
    version="1.1.0",
)


class ScoreIn(BaseModel):
    features: dict[str, float] = Field(..., description="Time, V1..V28, Amount (raw rupees/euros)")
    explain: bool = False


class ReviewIn(BaseModel):
    id: int
    decision: str  # confirm_fraud | false_alarm | needs_review
    note: str = ""


class CaseIn(BaseModel):
    features: dict[str, float | None] = Field(
        ...,
        description="Time, V1..V28, Amount. Null or non-finite values abstain instead of scoring.",
    )
    live: bool | None = Field(
        default=None,
        description="True calls Grok. False is the scripted clerk. Omit to call Grok only when XAI_API_KEY is set.",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "generator_in_path": False,
        "model": "xgb_gan",
        "case_agent": True,
        "llm": XAI_MODEL if os.environ.get("XAI_API_KEY") else "scripted",
    }


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "metrics.json missing")
    return json.loads(METRICS_PATH.read_text())


@app.post("/score")
def score(body: ScoreIn):
    names = feature_names()
    missing = [n for n in names if n not in body.features]
    if missing:
        raise HTTPException(400, f"missing features: {missing}")
    row = np.array([[body.features[n] for n in names]], dtype=np.float64)
    try:
        validate_finite(row)
        Xs, proba, thr = transform(row)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    p = float(proba[0])
    flag = p >= thr
    ood = ood_flags(Xs, names)[0]
    shap_top = None
    if body.explain:
        try:
            shap_top = shap_for_rows(Xs, top_k=8)[0]["top"]
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from e
    audit_id = log_score(
        p_fraud=p,
        flag=flag,
        threshold=thr,
        is_ood=ood["is_ood"],
        features={n: float(body.features[n]) for n in names},
        shap_top=shap_top,
    )
    return {
        "id": audit_id,
        "p_fraud": p,
        "flag": flag,
        "threshold": thr,
        "ood": ood,
        "shap": shap_top,
        "note": "generator is not used at inference",
    }


@app.post("/review")
def do_review(body: ReviewIn):
    try:
        review(body.id, body.decision, body.note)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"id": body.id, "decision": body.decision}


@app.get("/audit")
def audit(limit: int = 50):
    return {"rows": recent(limit)}


@app.post("/case")
def case(body: CaseIn):
    """Draft a cited review note. The agent cannot confirm fraud or clear it."""
    names = feature_names()
    missing = [name for name in names if name not in body.features]
    if missing:
        raise HTTPException(400, f"missing features: {missing}")
    if body.live and not os.environ.get("XAI_API_KEY"):
        raise HTTPException(503, "XAI_API_KEY is not set")
    try:
        return run_case(body.features, live=body.live)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
