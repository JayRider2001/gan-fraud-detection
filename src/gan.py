"""TensorFlow WGAN-GP: generator, critic, gradient penalty.

Generator loss = -mean(C(fake))
    The critic scores "how real". G wants that score high. We minimize, so we flip the sign.

Critic loss = mean(C(fake)) - mean(C(real)) + λ * GP
    Push fake scores down, real scores up, and keep the critic's slope near 1.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    C_HIDDEN,
    G_HIDDEN,
    GAN_BETA_1,
    GAN_BETA_2,
    GAN_LR,
    GP_LAMBDA,
    NOISE_DIM,
)


def build_generator(n_features: int, noise_dim: int = NOISE_DIM) -> keras.Model:
    """z ~ N(0,1)  ->  fake transaction row (same width as scaled features)."""
    model = keras.Sequential(name="generator")
    model.add(keras.Input(shape=(noise_dim,)))
    for i, units in enumerate(G_HIDDEN):
        model.add(layers.Dense(units, name=f"g_dense_{i}"))
        model.add(layers.LeakyReLU(0.2, name=f"g_lrelu_{i}"))
    model.add(layers.Dense(n_features, name="g_out"))  # linear: data is standardized
    return model


def build_critic(n_features: int) -> keras.Model:
    """Row -> unbounded score. No sigmoid. No batch-norm (it breaks GP)."""
    model = keras.Sequential(name="critic")
    model.add(keras.Input(shape=(n_features,)))
    for i, units in enumerate(C_HIDDEN):
        model.add(layers.Dense(units, name=f"c_dense_{i}"))
        model.add(layers.LeakyReLU(0.2, name=f"c_lrelu_{i}"))
    model.add(layers.Dense(1, name="c_out"))
    return model


def gradient_penalty(critic: keras.Model, real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
    """1-Lipschitz via (||grad C(x_hat)|| - 1)^2 on the line between real and fake."""
    batch = tf.shape(real)[0]
    eps = tf.random.uniform([batch, 1], 0.0, 1.0, dtype=real.dtype)
    mixed = eps * real + (1.0 - eps) * fake
    with tf.GradientTape() as tape:
        tape.watch(mixed)
        score = critic(mixed, training=True)
    grads = tape.gradient(score, mixed)
    grads = tf.reshape(grads, [batch, -1])
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    return tf.reduce_mean(tf.square(norm - 1.0))


def make_optimizers():
    # WGAN-GP paper: Adam with β1=0.
    g_opt = keras.optimizers.Adam(GAN_LR, beta_1=GAN_BETA_1, beta_2=GAN_BETA_2)
    c_opt = keras.optimizers.Adam(GAN_LR, beta_1=GAN_BETA_1, beta_2=GAN_BETA_2)
    return g_opt, c_opt


class WGANGP:
    def __init__(self, n_features: int, noise_dim: int = NOISE_DIM, gp_lambda: float = GP_LAMBDA):
        self.noise_dim = noise_dim
        self.gp_lambda = gp_lambda
        self.generator = build_generator(n_features, noise_dim)
        self.critic = build_critic(n_features)
        self.g_opt, self.c_opt = make_optimizers()

    @tf.function
    def critic_step(self, real: tf.Tensor) -> dict:
        batch = tf.shape(real)[0]
        noise = tf.random.normal((batch, self.noise_dim))
        with tf.GradientTape() as tape:
            fake = self.generator(noise, training=True)
            real_score = self.critic(real, training=True)
            fake_score = self.critic(fake, training=True)
            gp = gradient_penalty(self.critic, real, fake)
            loss = (
                tf.reduce_mean(fake_score)
                - tf.reduce_mean(real_score)
                + self.gp_lambda * gp
            )
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.c_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        return {
            "c_loss": loss,
            "gp": gp,
            "c_real": tf.reduce_mean(real_score),
            "c_fake": tf.reduce_mean(fake_score),
        }

    @tf.function
    def generator_step(self, batch: int) -> tf.Tensor:
        noise = tf.random.normal((batch, self.noise_dim))
        with tf.GradientTape() as tape:
            fake = self.generator(noise, training=True)
            score = self.critic(fake, training=True)
            # Maximize critic score on fakes == minimize the negative mean.
            loss = -tf.reduce_mean(score)
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        return loss

    def generate(self, n: int) -> tf.Tensor:
        noise = tf.random.normal((n, self.noise_dim))
        return self.generator(noise, training=False)
