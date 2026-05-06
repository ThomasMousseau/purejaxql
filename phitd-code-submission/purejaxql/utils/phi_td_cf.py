"""Characteristic-function utilities for distributional RL losses.

This module provides environment-agnostic JAX primitives used by multiple
training entrypoints (MoG, categorical, quantile, and related variants).
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from purejaxql.utils.cf_pareto import sample_truncated_pareto_alpha_1


def normalize_dist_loss_name(name: str) -> str:
    """Normalize config aliases for CTD ``DIST_LOSS``."""
    normalized = str(name).lower().replace("-", "_")
    if normalized in ("ce",):
        return "cross_entropy"
    return normalized


def build_categorical_cf(probs, support, omegas):
    """Characteristic function of a categorical / discrete distribution on fixed atoms.

    φ(ω) = Σ_a p_a · exp(i · ω · z_a) with atoms ``support[a] = z_a``.
    """
    phase = omegas[:, None] * support[None, :]
    kernels = jnp.exp(1j * phase)
    return jnp.einsum("...a,na->...n", probs, kernels, optimize=True)


def categorical_cf_weighted_mse(
    logits: jnp.ndarray,
    target_probs: jnp.ndarray,
    support: jnp.ndarray,
    omegas: jnp.ndarray,
    divide_by_omega_squared: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Categorical CF mismatch loss, optionally weighted by ``1 / omega^2``."""
    pred_probs = jax.nn.softmax(logits, axis=-1)
    stop_grad = jax.lax.stop_gradient
    phi_pred = build_categorical_cf(pred_probs, support, omegas)
    phi_target = build_categorical_cf(stop_grad(target_probs), support, omegas)
    complex_error = phi_pred - phi_target
    squared_error = jnp.abs(complex_error) ** 2
    imag_abs_error = jnp.abs(jnp.imag(complex_error))
    if divide_by_omega_squared:
        weight = (1.0 / jnp.maximum(omegas, 1e-8) ** 2).reshape(
            (1,) * (squared_error.ndim - 1) + (omegas.shape[0],)
        )
        squared_error = squared_error * weight
        imag_abs_error = imag_abs_error * weight
    return jnp.mean(squared_error), jnp.mean(imag_abs_error)


def build_dirac_mixture_cf(pi, mu, omegas):
    """Characteristic function of a mixture of Dirac masses."""
    pi_e = pi[..., None, :, :]
    mu_e = mu[..., None, :, :]
    w = omegas[..., None, None]
    phase = w * mu_e
    comp_cf = jnp.exp(1j * phase)
    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def build_quantile_cf(quantiles, omegas):
    """Characteristic function of the empirical / quantile-histogram distribution."""

    def for_omega(w: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.exp(1j * w * quantiles), axis=-1)

    stacked = jax.vmap(for_omega)(omegas)
    return jnp.moveaxis(stacked, 0, -1)


def build_mog_cf(pi, mu, sigma, omegas):
    """Build the Mixture-of-Gaussians characteristic function."""
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
    """Closed-form Q from mixture parameters: Q(s, a) = E[G] = Σ_k π_k · μ_k."""
    return jnp.sum(pi * mu, axis=-1)


def build_cauchy_mixture_cf(pi, loc, scale, omegas):
    """Characteristic function of a mixture of Cauchy(loc_k, scale_k)."""
    pi_e = pi[..., None, :, :]
    loc_e = loc[..., None, :, :]
    scale_e = scale[..., None, :, :]
    w = omegas[..., None, None]
    phase = w * loc_e
    decay = -scale_e * jnp.abs(w)
    comp_cf = jnp.exp(decay + 1j * phase)
    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def build_gamma_mixture_cf(pi, shape, scale, omegas):
    """Characteristic function of a mixture of Gamma(shape_k, scale_k)."""
    pi_e = pi[..., None, :, :]
    shape_e = shape[..., None, :, :]
    scale_e = scale[..., None, :, :]
    w = omegas[..., None, None]
    z = (1.0 - 1j * scale_e * w).astype(jnp.complex64)
    comp_cf = jnp.power(z, -shape_e)
    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def gamma_mixture_q_values(pi, shape, scale):
    """Expectation Q(s, a) = E[G] = Σ_k π_k · α_k · θ_k for Gamma components."""
    return jnp.sum(pi * shape * scale, axis=-1)


def build_laplace_mixture_cf(pi, loc, scale, omegas):
    """Characteristic function of a mixture of Laplace(location μ, scale b)."""
    pi_e = pi[..., None, :, :]
    loc_e = loc[..., None, :, :]
    scale_e = scale[..., None, :, :]
    w = omegas[..., None, None]
    denom = 1.0 + (scale_e * w) ** 2
    comp_cf = jnp.exp(1j * loc_e * w) / denom
    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def build_logistic_mixture_cf(pi, loc, scale, omegas):
    """Characteristic function of a mixture of logistic distributions."""
    pi_e = pi[..., None, :, :]
    loc_e = loc[..., None, :, :]
    scale_e = scale[..., None, :, :]
    w = omegas[..., None, None]
    x = jnp.pi * scale_e * w
    sinh_x = jnp.sinh(x)
    # Stable limit x/sinh(x) -> 1 as x -> 0.
    ratio_x_over_sinh_x = jnp.where(
        jnp.abs(x) > 1e-7, x / sinh_x, jnp.ones_like(x)
    )
    comp_cf = jnp.exp(1j * loc_e * w) * ratio_x_over_sinh_x
    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def sample_frequencies(
    key,
    num_samples,
    omega_max,
    scale=None,
    distribution="half_laplacian",
    gaussian_mean=0.0,
    gaussian_std=1.0,
    omega_min=None,
):
    """Sample non-negative frequencies for CF-based losses."""
    distribution = str(distribution).lower()
    if distribution == "half_laplace":
        distribution = "half_laplacian"

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
        # Truncated half-normal inverse CDF on [0, omega_max].
        u = jax.random.uniform(key, shape=(num_samples,))
        cdf_max = jsp.erf(omega_max / (gaussian_std * jnp.sqrt(2.0)))
        cdf_max = jnp.clip(cdf_max, 1e-8, 1.0 - 1e-8)
        omega = gaussian_std * jnp.sqrt(2.0) * jsp.erfinv(u * cdf_max)
        return jnp.clip(omega, 0.0, omega_max)

    if distribution in ("pareto_1", "pareto_alpha_1"):
        return sample_truncated_pareto_alpha_1(
            key,
            num_samples,
            omega_max,
            omega_min=omega_min,
        )

    raise ValueError(
        "Unknown distribution for sample_frequencies: "
        f"{distribution}. Supported: half_laplacian (alias half_laplace), pareto_1, uniform, half_gaussian."
    )
