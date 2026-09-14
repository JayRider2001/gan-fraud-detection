"""Train WGAN-GP on TRAIN-SET FRAUD ONLY. Val/test never enter this file."""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    CRITIC_PATH,
    G_STEPS,
    GAN_HISTORY,
    GENERATOR_PATH,
    N_CRITIC,
    NOISE_DIM,
    SEED,
)
from src.data import fraud_only, load_processed
from src.gan import WGANGP
from src.plots import plot_gan_losses


def set_seed(seed: int = SEED) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def make_dataset(fraud: np.ndarray) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(fraud)
    ds = ds.shuffle(buffer_size=len(fraud), reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train() -> dict:
    set_seed()
    data = load_processed()
    fraud = fraud_only(data["X_train"], data["y_train"])
    n_features = fraud.shape[1]
    print(f"[gan] training on {len(fraud)} train-set fraud rows, dim={n_features}")
    print(f"[gan] {G_STEPS} generator steps, n_critic={N_CRITIC}, batch={BATCH_SIZE}")

    wgan = WGANGP(n_features=n_features, noise_dim=NOISE_DIM)
    # Build weights
    _ = wgan.generator(tf.zeros((1, NOISE_DIM)))
    _ = wgan.critic(tf.zeros((1, n_features)))
    wgan.generator.summary()
    wgan.critic.summary()

    batches = iter(make_dataset(fraud))
    history = {k: [] for k in ("g_loss", "c_loss", "gp", "c_real", "c_fake")}

    for step in range(1, G_STEPS + 1):
        c_metrics = None
        for _ in range(N_CRITIC):
            real = next(batches)
            c_metrics = wgan.critic_step(real)
        g_loss = wgan.generator_step(BATCH_SIZE)

        if step == 1 or step % 100 == 0 or step == G_STEPS:
            rec = {
                "g_loss": float(g_loss.numpy()),
                "c_loss": float(c_metrics["c_loss"].numpy()),
                "gp": float(c_metrics["gp"].numpy()),
                "c_real": float(c_metrics["c_real"].numpy()),
                "c_fake": float(c_metrics["c_fake"].numpy()),
            }
            for k, v in rec.items():
                history[k].append((step, v))
            print(
                f"[gan] step {step:5d}/{G_STEPS}  "
                f"g_loss={rec['g_loss']:+.3f}  "
                f"c_loss={rec['c_loss']:+.3f}  "
                f"gp={rec['gp']:.3f}  "
                f"C(real)={rec['c_real']:+.3f}  "
                f"C(fake)={rec['c_fake']:+.3f}"
            )

    wgan.generator.save(GENERATOR_PATH)
    wgan.critic.save(CRITIC_PATH)
    GAN_HISTORY.write_text(json.dumps(history, indent=2))
    plot_gan_losses(history)
    print(f"[gan] saved {GENERATOR_PATH.name}, {CRITIC_PATH.name}")
    return history


if __name__ == "__main__":
    train()
