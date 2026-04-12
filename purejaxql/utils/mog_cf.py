"""
Mixture-of-Gaussians characteristic-function helpers (MoG–CF / “no-collapse” CVI).

These are the same pure-JAX primitives used in the CleanRL single-file scripts
(``old_cvi/cvi_utils_nocollapse_jax.py``): they are **environment-agnostic** and
shared by MoG-DQN, MoG-PQN, and MoG-PQN-RNN. The purejaxql training entrypoints
import them here so the Bellman/CF math stays identical while the *training loop*
follows this repo’s Flax + Hydra + vector-env layout.
"""

import jax
import jax.numpy as jnp


def build_mog_cf(pi, mu, sigma, omegas):
    """Construct the Mixture-of-Gaussians characteristic function.

    φ(ω) = Σ_k π_k · exp(i·ω·μ_k − ½·ω²·σ²_k)

    This CF is **always valid** by construction:
      - φ(0) = Σ π_k = 1  (since π comes from softmax)
      - |φ(ω)| ≤ Σ π_k |exp(...)| ≤ Σ π_k = 1
    And the Bellman operator maps MoG → MoG (closure).

    Args:
        pi:     (..., action_dim, M)  — mixture weights (sum to 1 over M).
        mu:     (..., action_dim, M)  — component means.
        sigma:  (..., action_dim, M)  — component std devs (positive).
        omegas: (N,) — sampled frequencies (positive, exploiting conjugate symmetry).

    Returns:
        Complex array (..., N, action_dim) — CF values at each frequency.
    """
    pi_e = pi[..., None, :, :]
    mu_e = mu[..., None, :, :]
    sigma_e = sigma[..., None, :, :]
    w = omegas[..., None, None]

    phase = w * mu_e
    decay = -0.5 * w**2 * sigma_e**2

    comp_cf = jnp.exp(decay + 1j * phase)

    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def mog_q_values(pi, mu):
    """Closed-form Q from MoG parameters: Q(s, a) = E[G] = Σ_k π_k · μ_k."""
    return jnp.sum(pi * mu, axis=-1)


def sample_frequencies(key, num_samples, omega_max, scale=None):
    """Sample ω from a truncated exponential (half-Laplacian) on (0, omega_max].

    Same sampling as the CleanRL CVI scripts: positive-only ω (conjugate symmetry)
    and exponential weighting toward small ω (replaces explicit FFT frequency
    weighting).
    """
    if scale is None:
        scale = omega_max / 3.0

    u = jax.random.uniform(key, shape=(num_samples,))
    max_cdf = 1.0 - jnp.exp(-omega_max / scale)
    omega = -scale * jnp.log(1.0 - u * max_cdf)
    return omega
