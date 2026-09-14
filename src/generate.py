"""Sample synthetic fraud from the trained generator (scaled feature space)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.config import GENERATOR_PATH, SEED, SYNTH_MULTIPLIER, SYNTH_PATH
from src.data import fraud_only, load_processed


def n_synth_for(n_real_fraud: int) -> int:
    return int(SYNTH_MULTIPLIER * n_real_fraud)


def generate(n: int | None = None) -> np.ndarray:
    data = load_processed()
    real_fraud = fraud_only(data["X_train"], data["y_train"])
    n = n if n is not None else n_synth_for(len(real_fraud))
    print(f"[generate] sampling {n} fakes from {GENERATOR_PATH.name}")

    generator = tf.keras.models.load_model(GENERATOR_PATH)
    noise_dim = generator.input_shape[-1]
    tf.keras.utils.set_random_seed(SEED)
    noise = tf.random.normal((n, noise_dim))
    fake = generator(noise, training=False).numpy().astype(np.float32)
    np.save(SYNTH_PATH, fake)

    # Sanity: compare first moments on a couple of columns.
    print("[generate] mean |real fraud| vs |fake| (scaled space, first 6 cols):")
    for i in range(min(6, fake.shape[1])):
        print(
            f"    col {i:02d}  real={real_fraud[:, i].mean():+.3f}  "
            f"fake={fake[:, i].mean():+.3f}"
        )
    print(f"[generate] saved {SYNTH_PATH}")
    return fake


if __name__ == "__main__":
    generate()
