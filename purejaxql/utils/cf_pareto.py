"""Truncated Pareto (shape α=1) sampling for characteristic-function / Cramér losses.

For Pareto Type I with minimum scale ``x_m`` and shape α=1, the CDF on ``[x_m, ∞)`` is
``F(x) = 1 - x_m / x``. Truncating to ``[x_m, B]`` and sampling by inverse CDF gives::

    ω = x_m / (1 - u · (1 - x_m / B)),   u ~ Uniform(0, 1).

Using these ω with **unweighted** squared CF error matches the intended Cramér objective;
pairing half-Laplacian ω with a separate ``1/ω²`` weight was a different estimator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_truncated_pareto_alpha_1(
    key: jax.Array,
    num_samples: int,
    omega_max: float | jnp.ndarray,
    omega_min: float | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sample ω from truncated Pareto (α=1) on ``[omega_min, omega_max]``.

    Args:
        key: PRNG key.
        num_samples: number of frequencies.
        omega_max: upper truncation ``B``.
        omega_min: lower bound ``x_m`` (scale). If None, uses ``max(1e-8, omega_max * 1e-6)``.

    Returns:
        (num_samples,) positive frequencies.
    """
    B = jnp.asarray(omega_max, dtype=jnp.float32)
    if omega_min is None:
        xm = jnp.maximum(jnp.float32(1e-8), B * jnp.float32(1e-6))
    else:
        xm = jnp.asarray(omega_min, dtype=jnp.float32)
    xm = jnp.maximum(xm, jnp.float32(1e-12))
    # Require xm < B for a valid truncated distribution.
    xm = jnp.minimum(xm, B * jnp.float32(1.0 - 1e-6))
    u = jax.random.uniform(key, shape=(num_samples,), dtype=jnp.float32)
    denom = 1.0 - u * (1.0 - xm / B)
    return xm / denom
