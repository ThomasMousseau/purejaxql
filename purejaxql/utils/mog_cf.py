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
import jax.scipy.special as jsp


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


def sample_frequencies(
    key,
    num_samples,
    omega_max,
    scale=None,
    distribution="half_laplacian",
    gaussian_mean=0.0,
    gaussian_std=1.0,
):
    """Sample positive ω frequencies for CF losses.

    Supported distributions:
      - ``half_laplacian``: truncated exponential on ``[0, omega_max]``.
      - ``uniform``: uniform on ``[0, omega_max]``.
      - ``half_gaussian``: absolute Gaussian ``|N(mean, std)|``, clipped to ``omega_max``.

    Notes:
      - ``half_laplacian`` remains the default for backward compatibility.
      - Frequencies are always non-negative (conjugate symmetry in CF space).
    """
    distribution = str(distribution).lower()

    if distribution == "half_laplacian":
        if scale is None:
            scale = omega_max / 3.0
        u = jax.random.uniform(key, shape=(num_samples,))
        max_cdf = 1.0 - jnp.exp(-omega_max / scale)
        return -scale * jnp.log(1.0 - u * max_cdf)

    if distribution == "uniform":
        return jax.random.uniform(
            key, shape=(num_samples,), minval=0.0, maxval=omega_max
        )

    if distribution == "half_gaussian":
        if gaussian_mean != 0.0:
            raise ValueError(
                "half_gaussian currently supports gaussian_mean=0.0 only."
            )
        if gaussian_std <= 0.0:
            raise ValueError("gaussian_std must be > 0 for half_gaussian.")
        # Truncated half-normal on [0, omega_max]:
        # F(x) = erf(x / (sigma * sqrt(2))) for x >= 0.
        # Inverse CDF sampling: x = sigma*sqrt(2)*erfinv(u * F(omega_max)).
        u = jax.random.uniform(key, shape=(num_samples,))
        cdf_max = jsp.erf(omega_max / (gaussian_std * jnp.sqrt(2.0)))
        cdf_max = jnp.clip(cdf_max, 1e-8, 1.0 - 1e-8)
        omega = gaussian_std * jnp.sqrt(2.0) * jsp.erfinv(u * cdf_max)
        return jnp.clip(omega, 0.0, omega_max)

    raise ValueError(
        "Unknown distribution for sample_frequencies: "
        f"{distribution}. Supported: half_laplacian, uniform, half_gaussian."
    )
