"""Analyst review UI: score a CSV, inspect SHAP, confirm or override.

Human-in-the-loop is a Top-Tier applied-AI pattern (healthcare / Azure DevOps refs).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.audit import log_score, recent, review
from src.case_agent import run_case
from src.explain import shap_for_rows
from src.features import feature_names, frame_to_raw, transform
from src.guardrails import ood_flags, validate_finite
from src.serving import load_bundle

st.set_page_config(page_title="Fraud review desk", layout="wide")
st.title("Fraud review desk")
st.caption("GAN-augmented XGBoost. The case agent drafts a cited note. A person confirms or clears it.")

names = feature_names()
_, _, _, thr = load_bundle()
st.sidebar.metric("val-chosen threshold", f"{thr:.3f}")
st.sidebar.write("Flag if P(fraud) ≥ threshold. Override is written to the audit log.")

tab_score, tab_audit = st.tabs(["Score batch", "Audit log"])

with tab_score:
    up = st.file_uploader("CSV with Time, V1–V28, Amount (raw)", type=["csv"])
    if up is None:
        st.info("Upload a slice of the Kaggle creditcard.csv, or any file with those columns.")
    else:
        df = pd.read_csv(up)
        raw, _ = frame_to_raw(df)
        validate_finite(raw)
        Xs, proba, thr = transform(raw)
        ood = ood_flags(Xs, names)
        out = df.copy()
        out["p_fraud"] = proba
        out["flag"] = (proba >= thr).astype(int)
        out["ood"] = [int(x["is_ood"]) for x in ood]
        st.dataframe(out[["p_fraud", "flag", "ood"] + [c for c in ("Amount", "Class") if c in out.columns]].head(50))
        st.write(f"flagged **{int(out['flag'].sum())}** / {len(out)}")

        idx = st.number_input("row index to explain", min_value=0, max_value=len(out) - 1, value=0)
        row_xs = Xs[int(idx) : int(idx) + 1]
        expl = shap_for_rows(row_xs, top_k=10)[0]
        st.subheader(f"SHAP for row {idx}  ·  P(fraud)={proba[int(idx)]:.4f}")
        shap_df = pd.DataFrame(expl["top"])
        st.dataframe(shap_df)

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        labels = [t["feature"] for t in expl["top"]][::-1]
        vals = [t["shap"] for t in expl["top"]][::-1]
        colors = ["#C0562A" if v > 0 else "#0F6C61" for v in vals]
        ax.barh(labels, vals, color=colors)
        ax.axvline(0, color="#1B365D", lw=0.8)
        ax.set_xlabel("SHAP (toward fraud if > 0)")
        st.pyplot(fig)
        plt.close(fig)

        audit_id = log_score(
            p_fraud=float(proba[int(idx)]),
            flag=bool(out.loc[int(idx), "flag"]),
            threshold=float(thr),
            is_ood=bool(ood[int(idx)]["is_ood"]),
            features={n: float(raw[int(idx), j]) for j, n in enumerate(names)},
            shap_top=expl["top"],
        )
        st.write(f"audit id **{audit_id}**")
        c1, c2, c3 = st.columns(3)
        if c1.button("Confirm fraud"):
            review(audit_id, "confirm_fraud")
            st.success("logged confirm_fraud")
        if c2.button("False alarm"):
            review(audit_id, "false_alarm")
            st.warning("logged false_alarm")
        if c3.button("Needs review"):
            review(audit_id, "needs_review")
            st.info("logged needs_review")

        if st.button("Draft case note"):
            features = {n: float(raw[int(idx), j]) for j, n in enumerate(names)}
            with st.spinner("Case agent is calling score, SHAP, and past reviews"):
                opened = run_case(features)
            st.subheader(f"Case {opened['audit_id']} · {opened['disposition']} · {opened['llm']}")
            if opened.get("llm_error"):
                st.warning(opened["llm_error"])
            st.write(
                f"p_fraud={opened['p_fraud']}  flag={opened['flag']}  "
                "The threshold owns the flag. This note does not."
            )
            note = opened.get("note") or {}
            for sentence in note.get("sentences") or []:
                cites = ", ".join(
                    f"{c.get('type')}:{c.get('feature', c.get('id', c.get('reason')))}"
                    for c in sentence["citations"]
                )
                st.markdown(f"- {sentence['text']}")
                st.caption(cites)
            st.dataframe(pd.DataFrame(opened["trace"]))

with tab_audit:
    rows = recent(100)
    if not rows:
        st.write("No scores yet.")
    else:
        st.dataframe(pd.DataFrame(rows).drop(columns=["features", "shap_top"], errors="ignore"))
