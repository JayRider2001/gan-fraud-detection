# GAN-augmented credit-card fraud detection

TensorFlow **WGAN-GP** that prints extra fraud rows, then **XGBoost** that actually scores transactions.

Fraud is 492 / 284,807 rows (**0.172%**). A model that always says “not fraud” is 99.83% accurate and catches nobody. The GAN is not the detector. It is a fake-fraud factory used only at training time.

```
CSV  →  stratified split  →  WGAN-GP on train fraud only  →  mix real+fake
                                                      ↘
                                        XGBoost  →  threshold on val  →  test once
```

At inference the generator is **not loaded**. Only the scaler and XGBoost run.

## Applied AI layer

The GAN is the training-time fraud factory. The thing an analyst runs is a system: a scoring API, SHAP, a review desk, an audit log, and a case agent that is only allowed to draft.

The agent has four tools and no others.

| tool | what it may do |
|---|---|
| `score_row` | run the shipped XGBoost. The validation threshold owns the flag. Tool arguments are ignored, so the model cannot hand in its own probability. |
| `explain_row` | SHAP top features for that audit id. Refused if the row was not scored or is out of distribution. |
| `similar_reviews` | cosine retrieval over **past analyst decisions** in the SQLite log, using the SHAP vector. |
| `draft_case_note` | a note whose every sentence cites a score id, a SHAP feature, a retrieved review, or an OOD feature the guardrail actually returned. |

`confirm_fraud` and `false_alarm` stay on `POST /review`. The agent cannot write them. `V1`–`V28` are PCA components, so a note cites the component and does not invent a merchant.

Non-finite rows produce no probability. Out-of-distribution rows are still scored (the number is real) and the note disposition is `abstain`, which means a person has to look before the flag is treated as a case decision.

With `XAI_API_KEY` set, the clerk is Grok (`grok-4.7`) through the xAI Responses API and those tools. The tool executor rejects an out-of-order call or a bad citation. If that call fails, the scripted clerk finishes the note and the response includes `llm_error`. With no key, or when the eval runs, the scripted clerk walks the same tools. `python run.py agent-eval` is offline: 30 cases, policy checks that illegal notes are rejected, a FastAPI smoke test, a recomputed test AUPRC that must match `metrics.json`, and a check that `src.gan` was never imported.

| component | where |
|---|---|
| REST API | `api/app.py` — `/score`, `/case`, `/review`, `/health`, `/metrics` |
| Case agent | `src/case_agent.py` |
| Eval | `src/agent_eval.py`, report in `artifacts/agent_eval.json` |
| Explainability | `src/explain.py` SHAP TreeExplainer |
| Human review | Streamlit `app/ui.py` and `POST /review` |
| Audit log | `src/audit.py` SQLite |
| Guardrails | `src/guardrails.py` — non-finite reject, OOD flags |
| Docker | `Dockerfile` — generator is not on the serve path |

```bash
curl -s localhost:8000/health
curl -s localhost:8000/case -H 'content-type: application/json' \
  -d '{"features": {"Time": 0, "V1": -1.3, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0, "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0, "Amount": 149.62}, "live": false}'
python run.py agent-eval
```

## Why WGAN-GP

Vanilla GAN training often dies (flat generator gradients, mode collapse). This repo implements Wasserstein GAN with gradient penalty in TensorFlow:

| piece | what it does |
|---|---|
| Generator `G` | noise `z` → fake transaction row |
| Critic `C` | row → score (not a probability, no sigmoid) |
| Critic loss | `mean(C(fake)) − mean(C(real)) + λ·GP` |
| Generator loss | `−mean(C(fake))`  (raise the critic’s score on fakes) |
| Gradient penalty | mix real and fake, push `‖∇C‖` toward 1 so the critic stays 1-Lipschitz |

`n_critic = 5`, `λ = 10`, Adam with `β1 = 0` (WGAN-GP defaults).

## Leakage protocol

1. Stratified 70 / 15 / 15 split **first**.
2. `StandardScaler` fit on **train only**.
3. GAN trained on **train-set frauds only**.
4. SMOTE / GAN oversampling happen on train only, to the **same** minority count.
5. Decision threshold chosen on **val** (max F1).
6. Test scored **once**, same test set for all three models.

## Three detectors, one XGBoost config

| model | training rows |
|---|---|
| `weighted` | raw train + `scale_pos_weight = n_neg / n_pos` |
| `smote` | SMOTE up to the same minority count |
| `gan` | real train + WGAN-GP fakes |

Primary metric: **AUPRC**. Accuracy is not reported as a win.

## Setup

Python **3.12** (TensorFlow has no 3.14 wheel). Dataset CSV is downloaded on first run (not stored in git).

```bash
git clone https://github.com/JayRider2001/gan-fraud-detection.git
cd gan-fraud-detection
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python run.py all
```

Or with pip: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python run.py all`

Stages if you want them separate:

```bash
python run.py data      # download + split + scale
python run.py gan       # train WGAN-GP, sample fakes
python run.py detect    # three XGBs, plots, metrics.json
python run.py infer --csv path/to/rows.csv
python run.py serve       # FastAPI on :8000  (POST /score, POST /case, POST /review)
python run.py ui          # Streamlit review desk on :8501
python run.py agent-eval  # offline case-agent eval; does not call Grok
```

```bash
docker build -t gan-fraud .
docker run --rm -p 8000:8000 gan-fraud
```

Dataset is pulled from TensorFlow’s public mirror of the ULB / Kaggle credit-card set (European cardholders, September 2013). `Time`, `V1`–`V28` (PCA), `Amount`. `Amount` is `log1p`’d then scaled.

## Results

Frozen **test** set (42,722 rows, **74 frauds**). Thresholds chosen on val (max F1), then applied once.

| model | AUPRC | ROC-AUC | precision | recall | F1 | FP | FN |
|---|---|---|---|---|---|---|---|
| XGB + class weights | 0.835 | 0.970 | 0.908 | 0.797 | 0.849 | 6 | 15 |
| SMOTE + XGB | 0.843 | 0.970 | 0.950 | 0.770 | 0.851 | 3 | 17 |
| **WGAN-GP + XGB** | **0.850** | **0.975** | **0.967** | 0.784 | **0.866** | **2** | 16 |

Primary metric is AUPRC. The GAN run is a modest lift, not a miracle: 74 test frauds is a noisy denominator, and on **validation** AUPRC the GAN was not first. We still keep the val-chosen threshold (no test-set fishing).

**Precision–recall (frozen test set)**

![Precision-recall curves](artifacts/plots/pr_curves.png)

**WGAN-GP training**

![GAN loss curves](artifacts/plots/gan_losses.png)

**Did the forger learn the fraud blob?** t-SNE of legit vs real fraud vs GAN fakes (scaled space). Synthetics overlap real fraud and sit apart from legit.

![t-SNE real vs fake](artifacts/plots/tsne.png)

Generator loss in code is the line from the crash guide:

```python
# src/gan.py  —  G wants C(fake) high; we minimize, so we flip the sign
loss = -tf.reduce_mean(score)
```

## Layout

```
src/config.py           hyperparameters
src/data.py             download, split, scale
src/gan.py              Generator, Critic, gradient_penalty, WGANGP
src/train_gan.py        5 critic steps / 1 generator step
src/generate.py         sample synthetic fraud
src/train_detector.py   weighted / SMOTE / GAN-aug XGBoost
src/evaluate.py         AUPRC, F1 threshold, confusion
src/infer.py            scaler + XGB only
src/explain.py          SHAP on the detector
src/audit.py            SQLite score, SHAP vector, case note, human override
src/guardrails.py       schema / OOD checks
src/case_agent.py       score / explain / retrieve / draft
src/agent_eval.py       offline eval of that agent
api/app.py              FastAPI
app/ui.py               Streamlit review desk
run.py                  CLI
Dockerfile              CPU serve image
```

## Limitations

- 492 frauds is a thin support for a GAN; fakes can memorize. t-SNE / feature means are the check.
- `V1`–`V28` are PCA — no merchant semantics, weak explainability.
- Two days in 2013, not a production stream. No concept drift.
- If GAN-aug does not beat SMOTE, that is a valid outcome and is reported as a comparison, not a fake SOTA.

