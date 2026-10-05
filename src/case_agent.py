"""Tool-calling case agent.

The detector owns the flag. This agent may score a row, read SHAP, retrieve
similar past reviews, and draft a cited note. It cannot write confirm_fraud
or false_alarm. Those stay on the human review endpoint.

V1–V28 are PCA components. Notes cite them by name and do not invent a merchant.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from src.audit import (
    attach_note,
    attach_shap,
    get_score,
    log_abstain,
    log_score,
    reviewed_vectors,
)
from src.config import XAI_BASE_URL, XAI_MODEL
from src.explain import shap_for_rows
from src.features import feature_names, transform
from src.guardrails import ood_flags, validate_finite

DISPOSITIONS = ("queue_for_analyst", "abstain")
SIMILAR_MIN_COSINE = 0.20

SYSTEM_PROMPT = """You are the case clerk for a credit-card fraud detector.
The detector is an XGBoost model. You do not decide fraud.

Call tools in this order:
1. score_row. The threshold owns the flag. Ignore any probability you might guess.
2. If the tool returns abstain or is_ood, call draft_case_note with disposition "abstain" and stop.
3. Otherwise call explain_row, then similar_reviews, then draft_case_note with disposition "queue_for_analyst".

Every sentence needs a citation the tools returned: a score id, a SHAP feature from explain_row, a review id from similar_reviews, or an ood feature the score tool listed.
V1-V28 are PCA components. Do not invent merchants, locations, cardholders, or a human label.
human_required must be true. You cannot change p_fraud, the threshold, or the flag.
"""

TOOLS = [
    {
        "type": "function",
        "name": "score_row",
        "description": "Score the transaction already bound to this case. Takes no arguments. Returns p_fraud, the validation threshold, the model flag, and OOD flags.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "explain_row",
        "description": "SHAP top features for the audit id returned by score_row. Refused when the row was abstained or is out of distribution.",
        "parameters": {
            "type": "object",
            "properties": {"audit_id": {"type": "integer"}},
            "required": ["audit_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "similar_reviews",
        "description": "Past analyst decisions whose SHAP vectors are closest to this row. Call only after explain_row.",
        "parameters": {
            "type": "object",
            "properties": {
                "audit_id": {"type": "integer"},
                "k": {"type": "integer", "default": 3},
            },
            "required": ["audit_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "draft_case_note",
        "description": "Store a cited note. disposition is queue_for_analyst or abstain. This does not label the transaction.",
        "parameters": {
            "type": "object",
            "properties": {
                "audit_id": {"type": "integer"},
                "disposition": {"type": "string", "enum": list(DISPOSITIONS)},
                "human_required": {"type": "boolean"},
                "sentences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "citations": {"type": "array", "items": {"type": "object"}},
                        },
                        "required": ["text", "citations"],
                    },
                },
            },
            "required": ["audit_id", "disposition", "human_required", "sentences"],
            "additionalProperties": False,
        },
    },
]


class CaseState:
    def __init__(self, features: dict, db_path: Path | None):
        self.features = dict(features)
        self.db_path = db_path
        self.names = feature_names()
        self.trace: list[dict] = []
        self.xs: np.ndarray | None = None
        self.scored: dict | None = None
        self.explained: dict | None = None
        self.similar: list | None = None
        self.note: dict | None = None
        self.status: str | None = None
        self.llm = "scripted"
        self.llm_error: str | None = None
        self.draft_fallback = False

    @property
    def audit_id(self) -> int | None:
        if self.scored is None:
            return None
        return int(self.scored["audit_id"])


def _as_float(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        number = float(value)
        if np.isfinite(number):
            return number
    return None


def _mentioned_features(text: str, names: list[str]) -> list[str]:
    found = []
    remaining = text
    for name in sorted(names, key=len, reverse=True):
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, remaining):
            found.append(name)
            remaining = re.sub(pattern, " ", remaining)
    return found


def _tool_ok(state: CaseState, name: str) -> bool:
    return any(step["tool"] == name and step["ok"] for step in state.trace)


def validate_note(state: CaseState, note: dict) -> list[str]:
    """Reject a draft that cites evidence the tools did not return, or that labels the row."""
    errors = []
    if not isinstance(note, dict):
        return ["note must be an object"]
    extra = set(note) - {"audit_id", "disposition", "human_required", "sentences"}
    if extra:
        errors.append(f"unknown note fields: {sorted(extra)}")
    if note.get("human_required") is not True:
        errors.append("human_required must be true")
    disposition = note.get("disposition")
    if disposition not in DISPOSITIONS:
        errors.append("disposition must be queue_for_analyst or abstain")
    if state.scored is None:
        errors.append("score_row has not succeeded")
        return errors
    if note.get("audit_id") != state.audit_id:
        errors.append("audit_id does not match the scored row")

    ood = bool(state.scored.get("ood", {}).get("is_ood"))
    abstained = state.status == "abstain"
    if abstained or ood:
        if disposition != "abstain":
            errors.append("out-of-distribution and non-finite rows require disposition abstain")
    elif disposition != "queue_for_analyst":
        errors.append("a scored in-distribution row uses disposition queue_for_analyst")
    if not abstained and not ood and state.explained is None:
        errors.append("explain_row has not succeeded")
    if not abstained and not ood and state.similar is None:
        errors.append("similar_reviews has not succeeded")

    sentences = note.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        errors.append("sentences must be a non-empty list")
        return errors

    allowed_shap = set()
    if state.explained is not None:
        allowed_shap = {item["feature"] for item in state.explained["top"]}
    allowed_ood = set(state.scored.get("ood", {}).get("ood_features") or [])
    allowed_reviews = set()
    if state.similar:
        allowed_reviews = {int(item["id"]) for item in state.similar}

    saw_score = False
    saw_shap = False
    for index, sentence in enumerate(sentences):
        if not isinstance(sentence, dict):
            errors.append(f"sentence {index} is not an object")
            continue
        text = sentence.get("text") or ""
        if not str(text).strip():
            errors.append(f"sentence {index} is empty")
        citations = sentence.get("citations")
        if not isinstance(citations, list) or not citations:
            errors.append(f"sentence {index} has no citation")
            citations = []
        cited_features = set()
        for citation in citations:
            if not isinstance(citation, dict):
                errors.append(f"sentence {index} has a citation that is not an object")
                continue
            kind = citation.get("type")
            if kind == "score":
                if citation.get("id") != state.audit_id:
                    errors.append(f"sentence {index} cites a score that was not produced")
                else:
                    saw_score = True
            elif kind == "audit":
                if citation.get("id") != state.audit_id:
                    errors.append(f"sentence {index} cites a different audit id")
            elif kind == "shap":
                feature = citation.get("feature")
                if feature not in allowed_shap:
                    errors.append(f"sentence {index} cites SHAP feature {feature} outside the top-k")
                else:
                    saw_shap = True
                    cited_features.add(feature)
            elif kind == "ood_feature":
                feature = citation.get("feature")
                if feature not in allowed_ood:
                    errors.append(f"sentence {index} cites OOD feature {feature} the guardrail did not flag")
                else:
                    cited_features.add(feature)
            elif kind == "review":
                review_id = citation.get("id")
                if int(review_id) not in allowed_reviews:
                    errors.append(f"sentence {index} cites review {review_id} that similar_reviews did not return")
            elif kind == "guardrail":
                if citation.get("reason") != "non_finite" or not abstained:
                    errors.append(f"sentence {index} cites a guardrail that did not fire")
            else:
                errors.append(f"sentence {index} has unknown citation type {kind}")
        for feature in _mentioned_features(str(text), state.names):
            if feature not in cited_features:
                errors.append(f"sentence {index} mentions {feature} without citing it")
        if abstained and re.search(r"\d+\.\d+", str(text)):
            errors.append(f"sentence {index} states a number on a row that was not scored")

    if not abstained and not saw_score:
        errors.append("note never cites the score id")
    if not abstained and not ood and not saw_shap:
        errors.append("note never cites a SHAP feature")
    return errors


def build_note(state: CaseState) -> dict:
    """Deterministic cited note. The live model may draft its own; this one always validates."""
    scored = state.scored
    assert scored is not None
    audit_id = int(scored["audit_id"])
    if state.status == "abstain":
        return {
            "audit_id": audit_id,
            "disposition": "abstain",
            "human_required": True,
            "sentences": [
                {
                    "text": (
                        "The row has a non-finite value, so no fraud probability was produced. "
                        "A human has to review the raw input."
                    ),
                    "citations": [
                        {"type": "guardrail", "reason": "non_finite"},
                        {"type": "audit", "id": audit_id},
                    ],
                }
            ],
        }
    if scored["ood"]["is_ood"]:
        feats = list(scored["ood"]["ood_features"][:4])
        shown = ", ".join(feats) if feats else "a scaled feature"
        return {
            "audit_id": audit_id,
            "disposition": "abstain",
            "human_required": True,
            "sentences": [
                {
                    "text": (
                        f"Audit {audit_id} is out of distribution on {shown}. "
                        "A human has to review it. This note is not a case decision."
                    ),
                    "citations": [{"type": "score", "id": audit_id}]
                    + [{"type": "ood_feature", "feature": name} for name in feats],
                }
            ],
        }

    assert state.explained is not None
    sentences = [
        {
            "text": (
                f"Detector score {scored['p_fraud']:.4f} against validation threshold "
                f"{scored['threshold']:.4f}; model flag is {int(scored['flag'])}. "
                f"A person still has to review audit {audit_id}."
            ),
            "citations": [{"type": "score", "id": audit_id}],
        }
    ]
    for item in state.explained["top"][:2]:
        direction = "toward fraud" if item["shap"] > 0 else "toward legitimate"
        sentences.append(
            {
                "text": (
                    f"PCA component {item['feature']} is a cited contributor "
                    f"({item['shap']:+.4f}, {direction})."
                ),
                "citations": [{"type": "shap", "feature": item["feature"]}],
            }
        )
    if state.similar:
        neighbor = state.similar[0]
        if neighbor["cosine"] >= SIMILAR_MIN_COSINE:
            sentences.append(
                {
                    "text": (
                        f"Nearest reviewed audit {neighbor['id']} was marked {neighbor['review']} "
                        f"at SHAP-vector cosine {neighbor['cosine']:.2f}."
                    ),
                    "citations": [{"type": "review", "id": int(neighbor["id"])}],
                }
            )
    return {
        "audit_id": audit_id,
        "disposition": "queue_for_analyst",
        "human_required": True,
        "sentences": sentences,
    }


def _finish_score(state: CaseState, payload: dict) -> dict:
    state.scored = payload
    state.status = payload["status"]
    return {"ok": True, **payload}


def tool_score(state: CaseState, _args: dict) -> dict:
    # Arguments are ignored on purpose. The model cannot supply a probability or a different row.
    if state.scored is not None:
        return {"ok": True, **state.scored}

    missing = [name for name in state.names if name not in state.features]
    if missing:
        return {"ok": False, "error": f"missing features: {missing}"}

    cleaned: dict = {}
    values = []
    non_finite = False
    for name in state.names:
        number = _as_float(state.features[name])
        if number is None:
            non_finite = True
            cleaned[name] = None
        else:
            cleaned[name] = number
            values.append(number)

    if non_finite:
        audit_id = log_abstain(cleaned, "non_finite", db_path=state.db_path)
        payload = {
            "audit_id": audit_id,
            "status": "abstain",
            "reason": "non_finite",
            "p_fraud": None,
            "flag": None,
            "threshold": None,
            "ood": {"is_ood": False, "n_ood_features": 0, "ood_features": []},
            "label_owner": "threshold",
            "agent_may_relabel": False,
        }
        return _finish_score(state, payload)

    row = np.array([values], dtype=np.float64)
    validate_finite(row)
    xs, proba, threshold = transform(row)
    probability = float(proba[0])
    flag = bool(probability >= threshold)
    ood = ood_flags(xs, state.names)[0]
    audit_id = log_score(
        p_fraud=probability,
        flag=flag,
        threshold=threshold,
        is_ood=ood["is_ood"],
        features=cleaned,
        db_path=state.db_path,
    )
    state.xs = xs
    payload = {
        "audit_id": audit_id,
        "status": "scored",
        "p_fraud": probability,
        "flag": flag,
        "threshold": float(threshold),
        "ood": ood,
        "label_owner": "threshold",
        "agent_may_relabel": False,
    }
    return _finish_score(state, payload)


def tool_explain(state: CaseState, args: dict) -> dict:
    if state.scored is None:
        return {"ok": False, "error": "call score_row before explain_row"}
    if state.status == "abstain":
        return {"ok": False, "error": "row was not scored; draft an abstain note"}
    if state.scored["ood"]["is_ood"]:
        return {
            "ok": False,
            "error": "row is out of distribution; call draft_case_note with disposition abstain",
        }
    if args.get("audit_id") != state.audit_id:
        return {"ok": False, "error": "audit_id does not match the scored row"}
    if state.explained is not None:
        return {"ok": True, "audit_id": state.audit_id, "top": state.explained["top"]}
    explained = shap_for_rows(state.xs, top_k=8)[0]
    state.explained = explained
    attach_shap(state.audit_id, explained["top"], explained["vector"], db_path=state.db_path)
    return {"ok": True, "audit_id": state.audit_id, "top": explained["top"]}


def tool_similar(state: CaseState, args: dict) -> dict:
    if state.explained is None:
        return {"ok": False, "error": "call explain_row before similar_reviews"}
    if args.get("audit_id") != state.audit_id:
        return {"ok": False, "error": "audit_id does not match the scored row"}
    k = int(args.get("k") or 3)
    k = min(max(k, 1), 5)
    query = np.asarray(state.explained["vector"], dtype=np.float64)
    query_norm = float(np.linalg.norm(query))
    ranked = []
    for row in reviewed_vectors(state.audit_id, db_path=state.db_path):
        vector = np.asarray(row["shap_vector"], dtype=np.float64)
        denom = query_norm * float(np.linalg.norm(vector))
        cosine = float(np.dot(query, vector) / denom) if denom else 0.0
        ranked.append(
            {
                "id": int(row["id"]),
                "review": row["review"],
                "cosine": round(cosine, 4),
                "p_fraud": row["p_fraud"],
                "flag": row["flag"],
            }
        )
    ranked.sort(key=lambda item: item["cosine"], reverse=True)
    state.similar = ranked[:k]
    return {"ok": True, "audit_id": state.audit_id, "reviews": state.similar}


def tool_draft(state: CaseState, args: dict) -> dict:
    if state.scored is None:
        return {"ok": False, "error": "call score_row before draft_case_note"}
    ood = bool(state.scored.get("ood", {}).get("is_ood"))
    if state.status != "abstain" and not ood:
        if state.explained is None:
            return {"ok": False, "error": "call explain_row before draft_case_note"}
        if state.similar is None:
            return {"ok": False, "error": "call similar_reviews before draft_case_note"}
    errors = validate_note(state, args)
    if errors:
        return {"ok": False, "error": "note rejected", "details": errors}
    stored = {
        "audit_id": state.audit_id,
        "disposition": args["disposition"],
        "human_required": True,
        "sentences": args["sentences"],
    }
    attach_note(state.audit_id, stored, db_path=state.db_path)
    state.note = stored
    return {"ok": True, "audit_id": state.audit_id, "disposition": stored["disposition"]}


_TOOLS = {
    "score_row": tool_score,
    "explain_row": tool_explain,
    "similar_reviews": tool_similar,
    "draft_case_note": tool_draft,
}


def execute(state: CaseState, name: str, args: dict | None) -> dict:
    """Run one tool. Illegal calls are recorded and do not change the label."""
    if name not in _TOOLS:
        result = {"ok": False, "error": f"unknown tool {name}"}
    else:
        try:
            result = _TOOLS[name](state, args or {})
        except ValueError as exc:
            result = {"ok": False, "error": str(exc)}
    state.trace.append(
        {
            "tool": name,
            "ok": bool(result.get("ok")),
            "error": None if result.get("ok") else result.get("error"),
        }
    )
    return result


def _next_scripted(state: CaseState) -> dict | None:
    if not _tool_ok(state, "score_row"):
        return {"name": "score_row", "arguments": {}}
    if state.status == "abstain" or (state.scored and state.scored["ood"]["is_ood"]):
        if state.note is None:
            return {"name": "draft_case_note", "arguments": build_note(state)}
        return None
    if not _tool_ok(state, "explain_row"):
        return {"name": "explain_row", "arguments": {"audit_id": state.audit_id}}
    if not _tool_ok(state, "similar_reviews"):
        return {"name": "similar_reviews", "arguments": {"audit_id": state.audit_id, "k": 3}}
    if state.note is None:
        return {"name": "draft_case_note", "arguments": build_note(state)}
    return None


def run_scripted(state: CaseState) -> None:
    for _ in range(6):
        call = _next_scripted(state)
        if call is None:
            return
        result = execute(state, call["name"], call["arguments"])
        if not result["ok"]:
            raise RuntimeError(f"scripted {call['name']} failed: {result.get('error')}")
    if state.note is None:
        raise RuntimeError("scripted agent produced no note")


def _xai_request(body: dict) -> dict:
    key = os.environ["XAI_API_KEY"]
    request = urllib.request.Request(
        f"{XAI_BASE_URL}/responses",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"xAI {exc.code}: {detail}") from exc


def _function_calls(payload: dict) -> list[dict]:
    calls = []
    for item in payload.get("output") or []:
        if item.get("type") == "function_call":
            calls.append(item)
    return calls


def run_live(state: CaseState) -> None:
    state.llm = XAI_MODEL
    body = {
        "model": XAI_MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "A transaction is already bound to the tools. "
                    "Review it. Do not estimate a probability yourself."
                ),
            },
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    previous = None
    for _ in range(8):
        if previous is not None:
            body = {
                "model": XAI_MODEL,
                "input": previous["outputs"],
                "tools": TOOLS,
                "previous_response_id": previous["id"],
            }
        payload = _xai_request(body)
        calls = _function_calls(payload)
        if not calls:
            return
        outputs = []
        for call in calls:
            raw_args = call.get("arguments") or "{}"
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            result = execute(state, call.get("name", ""), args)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(result),
                }
            )
        if state.note is not None:
            return
        previous = {"id": payload["id"], "outputs": outputs}


def _public(state: CaseState) -> dict:
    scored = state.scored or {}
    row = get_score(state.audit_id, db_path=state.db_path) if state.audit_id is not None else None
    return {
        "audit_id": state.audit_id,
        "status": state.status,
        "disposition": None if state.note is None else state.note["disposition"],
        "human_required": True,
        "p_fraud": scored.get("p_fraud"),
        "flag": scored.get("flag"),
        "threshold": scored.get("threshold"),
        "ood": scored.get("ood"),
        "shap": None if state.explained is None else state.explained["top"],
        "similar": state.similar or [],
        "note": state.note,
        "trace": state.trace,
        "llm": state.llm,
        "llm_error": state.llm_error,
        "draft_fallback": state.draft_fallback,
        "generator_in_path": False,
        "review": None if row is None else row.get("review"),
    }


def run_case(features: dict, db_path: Path | None = None, live: bool | None = None) -> dict:
    """Open one case. live=False forces the scripted clerk. live=None uses Grok when a key is set."""
    if live is None:
        live = bool(os.environ.get("XAI_API_KEY"))
    state = CaseState(features, db_path)
    if live:
        if not os.environ.get("XAI_API_KEY"):
            raise RuntimeError("XAI_API_KEY is not set")
        try:
            run_live(state)
        except Exception as exc:
            state.llm_error = str(exc)[:500]
        if state.note is None:
            state.draft_fallback = True
            state.llm = "scripted"
            run_scripted(state)
    else:
        run_scripted(state)
    return _public(state)
