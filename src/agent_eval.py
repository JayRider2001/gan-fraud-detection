"""Offline eval for the case agent.

Thirty scripted cases check tool order, citation faithfulness, and abstention.
A separate policy block feeds illegal notes to the same checker.
The GAN test AUPRC is recomputed from the shipped XGBoost and compared to metrics.json.
The generator is never imported.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.audit import get_score, review
from src.case_agent import CaseState, _function_calls, build_note, execute, run_case, validate_note
from src.config import AGENT_EVAL_PATH, METRICS_PATH, RAW_CSV, SEED
from src.data import load_processed
from src.features import feature_names, transform
from src.guardrails import ood_flags
from src.serving import load_bundle


def _features_from_row(row: pd.Series, names: list[str]) -> dict:
    return {name: float(row[name]) for name in names}


def _matrix(features: dict, names: list[str]) -> np.ndarray:
    return np.array([[features[name] for name in names]], dtype=np.float64)


def _not_ood(frame: pd.DataFrame, names: list[str], limit: int) -> list[int]:
    picked = []
    for idx in frame.index:
        features = _features_from_row(frame.loc[idx], names)
        xs, _, _ = transform(_matrix(features, names))
        if not ood_flags(xs, names)[0]["is_ood"]:
            picked.append(int(idx))
        if len(picked) == limit:
            break
    if len(picked) < limit:
        raise RuntimeError(f"needed {limit} in-distribution rows, found {len(picked)}")
    return picked


def _check_citations(state_result: dict) -> list[str]:
    """Re-check the stored note against the tool results that were actually returned."""
    errors = []
    note = state_result["note"]
    shap_features = {item["feature"] for item in (state_result.get("shap") or [])}
    review_ids = {int(item["id"]) for item in (state_result.get("similar") or [])}
    ood_features = set((state_result.get("ood") or {}).get("ood_features") or [])
    audit_id = state_result["audit_id"]
    for index, sentence in enumerate(note["sentences"]):
        if not sentence["citations"]:
            errors.append(f"sentence {index} has no citation")
        for citation in sentence["citations"]:
            kind = citation.get("type")
            if kind == "shap" and citation.get("feature") not in shap_features:
                errors.append(f"uncited-by-tool shap {citation.get('feature')}")
            elif kind == "review" and int(citation.get("id")) not in review_ids:
                errors.append(f"review {citation.get('id')} was not retrieved")
            elif kind == "score" and citation.get("id") != audit_id:
                errors.append("score citation mismatch")
            elif kind == "ood_feature" and citation.get("feature") not in ood_features:
                errors.append(f"ood citation {citation.get('feature')} was not flagged")
            elif kind == "audit" and citation.get("id") != audit_id:
                errors.append("audit citation mismatch")
    return errors


def _expected_tools(kind: str) -> list[str]:
    if kind in ("nan", "ood"):
        return ["score_row", "draft_case_note"]
    return ["score_row", "explain_row", "similar_reviews", "draft_case_note"]


def _eval_cases(db_path: Path) -> list[dict]:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"missing {RAW_CSV}")
    names = feature_names()
    frame = pd.read_csv(RAW_CSV)
    rng = np.random.RandomState(SEED)
    fraud_idx = rng.permutation(frame.index[frame["Class"] == 1].to_numpy())
    legit_idx = rng.permutation(frame.index[frame["Class"] == 0].to_numpy())
    fraud_frame = frame.loc[fraud_idx]
    legit_frame = frame.loc[legit_idx]

    seed_fraud = _not_ood(fraud_frame, names, 1)[0]
    seed_legit = _not_ood(legit_frame, names, 1)[0]
    happy_fraud = _not_ood(fraud_frame.drop(index=seed_fraud), names, 8)
    happy_legit = _not_ood(legit_frame.drop(index=seed_legit), names, 10)
    ood_base = _not_ood(legit_frame.drop(index=[seed_legit, *happy_legit]), names, 6)
    nan_base = list(legit_frame.drop(index=[seed_legit, *happy_legit, *ood_base]).index[:6])

    for idx, decision in ((seed_fraud, "confirm_fraud"), (seed_legit, "false_alarm")):
        opened = run_case(_features_from_row(frame.loc[idx], names), db_path=db_path, live=False)
        review(opened["audit_id"], decision, "eval seed", db_path=db_path)

    plan = (
        [("legit", idx) for idx in happy_legit]
        + [("fraud", idx) for idx in happy_fraud]
        + [("ood", idx) for idx in ood_base]
        + [("nan", idx) for idx in nan_base]
    )
    _, _, _, threshold = load_bundle()
    reports = []
    for number, (kind, idx) in enumerate(plan, start=1):
        features = _features_from_row(frame.loc[idx], names)
        if kind == "ood":
            features["V1"] = 1_000_000.0
            features["V2"] = -1_000_000.0
        if kind == "nan":
            features["Amount"] = float("nan")
        result = run_case(features, db_path=db_path, live=False)
        errors = []
        ok_tools = [step["tool"] for step in result["trace"] if step["ok"]]
        if ok_tools != _expected_tools(kind):
            errors.append(f"tool order {ok_tools}")
        if result["human_required"] is not True:
            errors.append("human_required is not true")
        if result["generator_in_path"]:
            errors.append("generator loaded")
        if result["draft_fallback"]:
            errors.append("scripted path used the live fallback")
        if result["llm"] != "scripted":
            errors.append(f"llm was {result['llm']}")
        if result["review"] is not None:
            errors.append("agent wrote a human review")
        stored = get_score(result["audit_id"], db_path=db_path)
        if stored is None or stored.get("review") is not None:
            errors.append("audit row has a human label")
        errors.extend(_check_citations(result))

        if kind == "nan":
            if result["disposition"] != "abstain" or result["p_fraud"] is not None:
                errors.append("non-finite row was scored")
            if result["flag"] is not None:
                errors.append("non-finite row has a flag")
        elif kind == "ood":
            if not result["ood"]["is_ood"] or result["disposition"] != "abstain":
                errors.append("OOD row was not abstained")
            if abs(result["threshold"] - threshold) > 1e-9:
                errors.append("threshold changed")
            xs, proba, _ = transform(_matrix(features, names))
            if abs(result["p_fraud"] - float(proba[0])) > 1e-9:
                errors.append("p_fraud does not match the detector")
            if result["flag"] != bool(float(proba[0]) >= threshold):
                errors.append("flag does not match the threshold")
            if "explain_row" in ok_tools:
                errors.append("OOD row was explained")
        else:
            if result["disposition"] != "queue_for_analyst":
                errors.append(f"disposition {result['disposition']}")
            if result["status"] != "scored":
                errors.append("happy path did not score")
            xs, proba, _ = transform(_matrix(features, names))
            if ood_flags(xs, names)[0]["is_ood"]:
                errors.append("happy-path row was OOD")
            if abs(result["p_fraud"] - float(proba[0])) > 1e-9:
                errors.append("p_fraud does not match the detector")
            if abs(result["threshold"] - threshold) > 1e-9:
                errors.append("threshold changed")
            if result["flag"] != bool(float(proba[0]) >= threshold):
                errors.append("flag does not match the threshold")
            if not result["shap"]:
                errors.append("missing SHAP")
        reports.append(
            {
                "n": number,
                "kind": kind,
                "audit_id": result["audit_id"],
                "disposition": result["disposition"],
                "p_fraud": result["p_fraud"],
                "flag": result["flag"],
                "tools": ok_tools,
                "pass": not errors,
                "errors": errors,
            }
        )
        print(f"[agent-eval] {number:02d}/30 {kind:5s} {'ok' if not errors else errors}", flush=True)
    return reports


def _policy_checks() -> list[dict]:
    """Illegal traces against the same executor. These do not score a real row."""
    names = feature_names()
    base = {name: 0.0 for name in names}
    checks = []

    state = CaseState(base, db_path=None)
    early = execute(state, "explain_row", {"audit_id": 1})
    checks.append(
        {
            "name": "explain_before_score",
            "pass": (not early["ok"]) and state.explained is None and state.note is None,
        }
    )

    state = CaseState(base, db_path=None)
    state.scored = {
        "audit_id": 7,
        "status": "scored",
        "p_fraud": 0.9,
        "flag": True,
        "threshold": 0.5,
        "ood": {"is_ood": False, "ood_features": []},
    }
    state.status = "scored"
    early_similar = execute(state, "similar_reviews", {"audit_id": 7})
    checks.append(
        {
            "name": "similar_before_explain",
            "pass": (not early_similar["ok"]) and state.similar is None,
        }
    )

    state.explained = {"top": [{"feature": "V14", "shap": 0.4, "value_scaled": 1.0}], "vector": [0.0] * 30}
    state.similar = []
    bad_feature = build_note(state)
    bad_feature["sentences"][1]["citations"] = [{"type": "shap", "feature": "V17"}]
    bad_feature["sentences"][1]["text"] = "PCA component V17 is a cited contributor (+0.4000, toward fraud)."
    rejected = validate_note(state, bad_feature)
    checks.append({"name": "shap_outside_topk", "pass": any("V17" in item for item in rejected)})

    relabel = build_note(state)
    relabel["disposition"] = "confirm_fraud"
    relabel_errors = validate_note(state, relabel)
    checks.append(
        {
            "name": "cannot_set_confirm_fraud",
            "pass": any("disposition" in item for item in relabel_errors),
        }
    )

    invented_review = build_note(state)
    invented_review["sentences"].append(
        {
            "text": "Nearest reviewed audit 99 was marked confirm_fraud at SHAP-vector cosine 0.99.",
            "citations": [{"type": "review", "id": 99}],
        }
    )
    review_errors = validate_note(state, invented_review)
    checks.append(
        {
            "name": "review_must_come_from_retrieval",
            "pass": any("review 99" in item for item in review_errors),
        }
    )

    sneak = build_note(state)
    sneak["p_fraud"] = 0.01
    sneak_errors = validate_note(state, sneak)
    checks.append({"name": "cannot_smuggle_p_fraud", "pass": any("p_fraud" in item for item in sneak_errors)})

    state.status = "abstain"
    state.scored = {
        "audit_id": 8,
        "status": "abstain",
        "p_fraud": None,
        "flag": None,
        "threshold": None,
        "ood": {"is_ood": False, "ood_features": []},
    }
    numbered = {
        "audit_id": 8,
        "disposition": "abstain",
        "human_required": True,
        "sentences": [
            {
                "text": "P(fraud) is 0.99 so this is fine.",
                "citations": [
                    {"type": "guardrail", "reason": "non_finite"},
                    {"type": "audit", "id": 8},
                ],
            }
        ],
    }
    number_errors = validate_note(state, numbered)
    checks.append(
        {
            "name": "abstain_cannot_state_a_probability",
            "pass": any("number" in item for item in number_errors),
        }
    )
    parsed = _function_calls(
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "score_row",
                    "arguments": "{}",
                    "call_id": "call_1",
                }
            ]
        }
    )
    checks.append(
        {
            "name": "responses_tool_call_parser",
            "pass": len(parsed) == 1 and parsed[0]["name"] == "score_row" and parsed[0]["call_id"] == "call_1",
        }
    )
    return checks


def _auprc_check() -> dict:
    data = load_processed()
    _, _, model, _ = load_bundle()
    proba = model.predict_proba(data["X_test"])[:, 1]
    recomputed = float(average_precision_score(data["y_test"], proba))
    recorded = float(json.loads(METRICS_PATH.read_text())["gan"]["test"]["auprc"])
    return {
        "recomputed_test_auprc": recomputed,
        "metrics_json_test_auprc": recorded,
        "abs_delta": abs(recomputed - recorded),
        "pass": abs(recomputed - recorded) < 1e-6,
    }


def _api_checks() -> list[dict]:
    from fastapi import HTTPException

    from api.app import CaseIn, case, health

    checks = []
    info = health()
    checks.append(
        {
            "name": "health",
            "pass": info["generator_in_path"] is False and info["case_agent"] is True and info["model"] == "xgb_gan",
        }
    )
    names = feature_names()
    try:
        case(CaseIn(features={"V1": 0.0}, live=False))
        missing_pass = False
    except HTTPException as exc:
        missing_pass = exc.status_code == 400
    checks.append({"name": "missing_features_400", "pass": missing_pass})

    finite = {name: 0.0 for name in names}
    finite["Amount"] = None
    abstained = case(CaseIn(features=finite, live=False))
    checks.append(
        {
            "name": "api_non_finite_abstain",
            "pass": abstained["disposition"] == "abstain" and abstained["p_fraud"] is None and abstained["review"] is None,
        }
    )

    ood = {name: 0.0 for name in names}
    ood["V1"] = 1_000_000.0
    ood["Amount"] = 12.5
    opened = case(CaseIn(features=ood, live=False))
    checks.append(
        {
            "name": "api_ood_abstain",
            "pass": (
                opened["disposition"] == "abstain"
                and opened["ood"]["is_ood"]
                and opened["human_required"] is True
                and opened["generator_in_path"] is False
                and opened["review"] is None
                and [step["tool"] for step in opened["trace"] if step["ok"]] == ["score_row", "draft_case_note"]
            ),
        }
    )
    return checks


def main() -> int:
    import sys

    policy = _policy_checks()
    api = _api_checks()
    auprc = _auprc_check()
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "agent_eval.db"
        cases = _eval_cases(db_path)
    policy = policy + api
    generator_loaded = any(name in sys.modules for name in ("src.gan", "src.train_gan", "src.generate"))
    report = {
        "n_cases": len(cases),
        "cases_passed": sum(1 for case in cases if case["pass"]),
        "policy_passed": sum(1 for check in policy if check["pass"]),
        "policy_total": len(policy),
        "llm": "scripted",
        "generator_loaded": generator_loaded,
        "auprc": auprc,
        "cases": cases,
        "policy": policy,
    }
    report["pass"] = (
        report["n_cases"] == 30
        and report["cases_passed"] == 30
        and report["policy_passed"] == report["policy_total"]
        and auprc["pass"]
        and not generator_loaded
    )
    AGENT_EVAL_PATH.write_text(json.dumps(report, indent=2))
    print(
        f"[agent-eval] cases {report['cases_passed']}/{report['n_cases']}  "
        f"policy {report['policy_passed']}/{report['policy_total']}  "
        f"AUPRC {auprc['recomputed_test_auprc']:.6f} "
        f"(metrics {auprc['metrics_json_test_auprc']:.6f})  "
        f"generator_loaded={generator_loaded}"
    )
    print(f"[agent-eval] wrote {AGENT_EVAL_PATH}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
