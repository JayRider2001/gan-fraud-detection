"""Figures for the README: losses, PR curves, t-SNE, confusion matrices."""

from __future__ import annotations

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve

from src.config import PLOTS, SEED

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def plot_gan_losses(history: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    specs = [
        ("g_loss", "Generator loss  (−mean C(fake))", axes[0]),
        ("c_loss", "Critic loss", axes[1]),
        ("gp", "Gradient penalty", axes[2]),
    ]
    for key, title, ax in specs:
        steps, vals = zip(*history[key])
        ax.plot(steps, vals, color="#1B365D")
        ax.set_title(title)
        ax.set_xlabel("generator step")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS / "gan_losses.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plots] {path}")


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """curves[name] = (y_true, proba) on the same test set."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colors = {"weighted": "#4A5560", "smote": "#0F6C61", "gan": "#C0562A"}
    labels = {
        "weighted": "XGB + class weights",
        "smote": "SMOTE + XGB",
        "gan": "WGAN-GP + XGB",
    }
    from sklearn.metrics import average_precision_score

    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rec, prec, color=colors[name], lw=2, label=f"{labels[name]}  (AUPRC={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall on frozen test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS / "pr_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plots] {path}")


def plot_confusion(y_true, proba, threshold: float, title: str, filename: str) -> None:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    disp = ConfusionMatrixDisplay(cm, display_labels=["legit", "fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    path = PLOTS / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plots] {path}")


def plot_tsne(real_fraud: np.ndarray, fake_fraud: np.ndarray, legit: np.ndarray) -> None:
    rng = np.random.RandomState(SEED)
    n_legit = min(1500, len(legit))
    n_fake = min(1500, len(fake_fraud))
    n_real = min(len(real_fraud), 500)
    legit_s = legit[rng.choice(len(legit), n_legit, replace=False)]
    fake_s = fake_fraud[rng.choice(len(fake_fraud), n_fake, replace=False)]
    real_s = real_fraud[rng.choice(len(real_fraud), n_real, replace=False)]

    X = np.vstack([legit_s, real_s, fake_s])
    labels = (
        ["legit"] * n_legit
        + ["real fraud"] * n_real
        + ["GAN fake fraud"] * n_fake
    )
    print(f"[plots] t-SNE on {len(X)} points (this can take a minute)...")
    z = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=750,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    ).fit_transform(X)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    colors = {"legit": "#C5CDD6", "real fraud": "#0F6C61", "GAN fake fraud": "#C0562A"}
    for name in ("legit", "real fraud", "GAN fake fraud"):
        m = np.array(labels) == name
        ax.scatter(z[m, 0], z[m, 1], s=8, alpha=0.7, c=colors[name], label=name, linewidths=0)
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title("t-SNE: legit vs real fraud vs GAN fakes (scaled space)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    path = PLOTS / "tsne.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plots] {path}")
