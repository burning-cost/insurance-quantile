"""
GPD distribution utilities for EQRN.

Implements the Generalised Pareto Distribution in both standard (sigma, xi)
and orthogonal (nu, xi) parameterisations. The orthogonal parameterisation,
due to Pasche & Engelke (2024), uses nu = sigma * (xi + 1), which makes the
Fisher information matrix diagonal and improves gradient-based optimisation.

All functions operate on both numpy arrays and PyTorch tensors. Loss functions
return tensors; distribution utilities return numpy arrays.

Edge cases at xi ~ 0 (exponential limit) are handled throughout with Taylor
series or L'Hopital expansions to avoid 0/0 numerical instability.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

# Type alias for values that can be either numpy or torch
ArrayLike = Union[np.ndarray, float]


# ---------------------------------------------------------------------------
# NumPy distribution utilities
# ---------------------------------------------------------------------------


def gpd_quantile(p: ArrayLike, xi: ArrayLike, sigma: ArrayLike, loc: ArrayLike = 0.0) -> np.ndarray:
    """Quantile function of the GPD (standard parameterisation).

    For a GPD with shape xi, scale sigma, and location loc, returns the value
    Q such that P(Y <= Q) = p.

    Parameters
    ----------
    p:
        Probability level(s), in (0, 1).
    xi:
        Shape parameter. Negative values give a bounded upper tail; positive
        values give a Pareto-like heavy tail. Constrained to > -1 for the
        mean to exist.
    sigma:
        Scale parameter. Must be positive.
    loc:
        Location (threshold shift). Defaults to 0.

    Returns
    -------
    np.ndarray
        Quantile values at probability p.

    Notes
    -----
    At xi = 0 (exponential), the quantile is loc - sigma * log(1 - p).
    The |xi| < 1e-8 branch uses this limit directly.
    """
    p = np.asarray(p, dtype=float)
    xi = np.asarray(xi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    loc = np.asarray(loc, dtype=float)

    result = np.where(
        np.abs(xi) < 1e-8,
        loc - sigma * np.log1p(-p),
        loc + sigma / xi * ((1.0 - p) ** (-xi) - 1.0),
    )
    return result


def gpd_survival(y: ArrayLike, xi: ArrayLike, sigma: ArrayLike, loc: ArrayLike = 0.0) -> np.ndarray:
    """Survival function P(Y > y) of the GPD.

    Parameters
    ----------
    y:
        Values at which to evaluate the survival function. Must be >= loc.
    xi, sigma, loc:
        GPD parameters (see gpd_quantile for descriptions).

    Returns
    -------
    np.ndarray
        Survival probabilities in [0, 1].
    """
    y = np.asarray(y, dtype=float)
    xi = np.asarray(xi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    loc = np.asarray(loc, dtype=float)

    z = (y - loc) / sigma
    result = np.where(
        np.abs(xi) < 1e-8,
        np.exp(-z),
        (1.0 + xi * z) ** (-1.0 / xi),
    )
    return np.clip(result, 0.0, 1.0)


def gpd_log_density(y: ArrayLike, xi: ArrayLike, sigma: ArrayLike, loc: ArrayLike = 0.0) -> np.ndarray:
    """Log density of the GPD: log f(y; xi, sigma, loc).

    Parameters
    ----------
    y, xi, sigma, loc:
        GPD parameters (see gpd_quantile).

    Returns
    -------
    np.ndarray
        Log density values. Returns -inf where y is outside the support.
    """
    y = np.asarray(y, dtype=float)
    xi = np.asarray(xi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    loc = np.asarray(loc, dtype=float)

    z = (y - loc) / sigma
    log_scale = -np.log(sigma)

    log_density = np.where(
        np.abs(xi) < 1e-8,
        log_scale - z,
        log_scale - (1.0 / xi + 1.0) * np.log1p(xi * z),
    )
    # Mask invalid support
    if xi >= 0:
        valid = z >= 0
    else:
        valid = (z >= 0) & (z <= -1.0 / xi)
    return np.where(valid, log_density, -np.inf)


def gpd_nll(y: ArrayLike, xi: ArrayLike, sigma: ArrayLike, loc: ArrayLike = 0.0) -> float:
    """Negative log-likelihood of the GPD for excess values y.

    Convenience wrapper around gpd_log_density for scalar summary.

    Parameters
    ----------
    y:
        Observed excesses. All must be >= 0 (i.e., y >= loc).
    xi, sigma, loc:
        GPD parameters.

    Returns
    -------
    float
        Total negative log-likelihood.
    """
    return float(-np.sum(gpd_log_density(y, xi, sigma, loc)))


def gpd_tvar(tau: ArrayLike, xi: ArrayLike, sigma: ArrayLike, loc: ArrayLike = 0.0) -> np.ndarray:
    """TVaR (Tail Value at Risk / Expected Shortfall) of the GPD.

    Computes E[Y | Y > Q_tau] where Q_tau is the tau-quantile.

    Parameters
    ----------
    tau:
        Probability level, in (0, 1). TVaR at tau means: expected value
        conditional on exceeding the tau-quantile.
    xi, sigma, loc:
        GPD parameters.

    Returns
    -------
    np.ndarray
        TVaR values.

    Raises
    ------
    ValueError
        If any xi >= 1, for which TVaR is infinite.

    Notes
    -----
    The closed form follows from the mean excess function of the GPD:
        TVaR(tau) = Q(tau) + (sigma + xi * (Q(tau) - loc)) / (1 - xi)
    This requires xi < 1.
    """
    tau = np.asarray(tau, dtype=float)
    xi = np.asarray(xi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    loc = np.asarray(loc, dtype=float)

    if np.any(xi >= 1.0):
        raise ValueError(
            "TVaR is infinite for xi >= 1. The EQRN output constraint "
            "limits xi < 0.7, so this should not occur with fitted models."
        )

    q_tau = gpd_quantile(tau, xi, sigma, loc)

    # For xi near 0, both branches converge; use the general formula
    tvar = q_tau + (sigma + xi * (q_tau - loc)) / (1.0 - xi)
    return tvar


def eqrn_quantile(
    tau: ArrayLike,
    tau_0: float,
    threshold: ArrayLike,
    xi: ArrayLike,
    sigma: ArrayLike,
) -> np.ndarray:
    """Conditional extreme quantile from the EQRN tail model.

    Inverts the GPD approximation above the intermediate threshold to get
    Q_x(tau) for tau > tau_0.

    Parameters
    ----------
    tau:
        Target quantile level(s). Must satisfy tau > tau_0.
    tau_0:
        Intermediate quantile level used during fitting.
    threshold:
        Per-observation threshold u(x) = Q_hat_x(tau_0).
    xi:
        Per-observation GPD shape parameter.
    sigma:
        Per-observation GPD scale parameter.

    Returns
    -------
    np.ndarray
        Conditional quantile Q_x(tau) per observation.

    Notes
    -----
    The formula is:
        Q_x(tau) = u(x) + sigma(x)/xi(x) * [((1 - tau_0)/(1 - tau))^xi(x) - 1]

    At xi = 0, L'Hopital gives:
        Q_x(tau) = u(x) + sigma(x) * log((1 - tau_0)/(1 - tau))
    """
    tau = np.asarray(tau, dtype=float)
    xi = np.asarray(xi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    threshold = np.asarray(threshold, dtype=float)

    ratio = (1.0 - tau_0) / (1.0 - tau)

    result = np.where(
        np.abs(xi) < 1e-6,
        threshold + sigma * np.log(ratio),
        threshold + sigma / xi * (ratio**xi - 1.0),
    )
    return result


def eqrn_tvar(
    tau: ArrayLike,
    tau_0: float,
    threshold: ArrayLike,
    xi: ArrayLike,
    sigma: ArrayLike,
) -> np.ndarray:
    """Conditional TVaR from the EQRN tail model.

    Computes E[Y | Y > Q_x(tau), X = x] using the GPD mean excess formula.

    Parameters
    ----------
    tau:
        Probability level for the tail event.
    tau_0, threshold, xi, sigma:
        See eqrn_quantile.

    Returns
    -------
    np.ndarray
        Conditional TVaR per observation.

    Notes
    -----
    Given Q_x(tau) = v, the excess above v has GPD with:
        sigma_v = sigma(x) + xi(x) * (v - u(x))
        same shape xi(x)

    So:
        TVaR_x(tau) = v + (sigma(x) + xi(x) * (v - u(x))) / (1 - xi(x))

    This requires xi < 1, which is guaranteed by the EQRN output activation.
    """
    q_tau = eqrn_quantile(tau, tau_0, threshold, xi, sigma)
    xi = np.asarray(xi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    threshold = np.asarray(threshold, dtype=float)

    excess_scale = sigma + xi * (q_tau - threshold)
    return q_tau + excess_scale / (1.0 - xi)


def eqrn_exceedance_prob(
    y_large: ArrayLike,
    tau_0: float,
    threshold: ArrayLike,
    xi: ArrayLike,
    sigma: ArrayLike,
) -> np.ndarray:
    """P(Y > y_large | X = x) from the EQRN tail approximation.

    Parameters
    ----------
    y_large:
        The threshold value(s) at which to evaluate exceedance probability.
    tau_0:
        Intermediate quantile level used during fitting.
    threshold, xi, sigma:
        Per-observation GPD parameters.

    Returns
    -------
    np.ndarray
        Exceedance probabilities P(Y > y_large | X = x).

    Notes
    -----
    Formula:
        P(Y > y | X = x) = (1 - tau_0) * [1 + xi(x) * (y - u(x)) / sigma(x)]^{-1/xi(x)}

    Values where y_large <= threshold return (1 - tau_0) as a lower bound
    (all exceedance mass above threshold).
    """
    y_large = np.asarray(y_large, dtype=float)
    xi = np.asarray(xi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    threshold = np.asarray(threshold, dtype=float)

    excess = np.maximum(y_large - threshold, 0.0)
    survival = np.where(
        np.abs(xi) < 1e-6,
        np.exp(-excess / sigma),
        (1.0 + xi * excess / sigma) ** (-1.0 / xi),
    )
    return (1.0 - tau_0) * np.clip(survival, 0.0, 1.0)


def eqrn_xl_layer(
    attachment: ArrayLike,
    limit: ArrayLike,
    tau_0: float,
    threshold: ArrayLike,
    xi: ArrayLike,
    sigma: ArrayLike,
    n_grid: int = 1000,
) -> np.ndarray:
    """Expected XL layer loss E[min(Y - attachment, limit)^+ | X = x].

    Prices the per-risk XL layer (attachment d, limit c) using the EQRN
    GPD approximation above the intermediate threshold.

    Parameters
    ----------
    attachment:
        XL attachment point d. Must be >= threshold for each observation.
    limit:
        XL limit c. The layer covers losses from d to d + c.
    tau_0:
        Intermediate quantile level used during fitting.
    threshold, xi, sigma:
        Per-observation GPD parameters.
    n_grid:
        Number of integration points for numerical quadrature over [d, d+c].

    Returns
    -------
    np.ndarray
        Expected layer loss per observation. Always >= 0.

    Notes
    -----
    Uses numerical integration:
        E[min(Y-d, c)^+] = integral_{d}^{d+c} P(Y > t | X=x) dt

    where P(Y > t | X = x) is evaluated using eqrn_exceedance_prob.
    """
    attachment = np.asarray(attachment, dtype=float)
    limit = np.asarray(limit, dtype=float)
    xi = np.asarray(xi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    threshold = np.asarray(threshold, dtype=float)

    # Integration grid: n_grid points over [attachment, attachment + limit]
    # Use trapezoid rule
    t = np.linspace(0.0, 1.0, n_grid)  # shape (n_grid,)
    # Broadcast: (n_obs,) + t * (n_obs,) = (n_obs, n_grid) after broadcast tricks
    # We iterate if obs is vectorised; for large n_obs use vectorised approach
    a = np.atleast_1d(attachment)
    lim = np.atleast_1d(limit)

    # Build integration grid: shape (n_obs, n_grid)
    t_grid = a[:, None] + lim[:, None] * t[None, :]  # (n_obs, n_grid)

    # Exceedance probabilities at each grid point
    p_grid = eqrn_exceedance_prob(
        t_grid,
        tau_0=tau_0,
        threshold=threshold[:, None] if threshold.ndim > 0 else threshold,
        xi=xi[:, None] if xi.ndim > 0 else xi,
        sigma=sigma[:, None] if sigma.ndim > 0 else sigma,
    )  # (n_obs, n_grid)

    # Integrate using trapezoid rule: integral ≈ (b-a) * mean of trapezia
    el = np.trapezoid(p_grid, t_grid, axis=1)
    return np.maximum(el, 0.0)


# ---------------------------------------------------------------------------
# PyTorch loss functions for training
# ---------------------------------------------------------------------------


def ogpd_loss_tensor(
    z: torch.Tensor,
    nu: torch.Tensor,
    xi: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Orthogonal GPD deviance loss for EQRN training.

    Implements the negative log-likelihood in the (nu, xi) orthogonal
    reparameterisation of Pasche & Engelke (2024):

        l_OGPD(z; nu, xi) = (1 + 1/xi) * log[1 + xi*(xi+1)*z/nu]
                           + log(nu) - log(xi + 1)

    Parameters
    ----------
    z:
        Excess values z_i = y_i - u(x_i) > 0. Shape (n_exceedances,).
    nu:
        Orthogonal scale parameter nu = sigma * (xi + 1). Shape (n_exceedances,).
        Must be > 0.
    xi:
        Shape parameter. Shape (n_exceedances,). Constrained to (-0.5, 0.7).
    reduction:
        'mean' (default), 'sum', or 'none'. Controls aggregation.

    Returns
    -------
    torch.Tensor
        Scalar loss (if reduction is 'mean' or 'sum') or per-sample losses.

    Notes
    -----
    Numerical stability:
    - When |xi| < 1e-5, the formula has 0/0 form. We use the exponential
      (xi = 0) limit: l(z; sigma, 0) = log(sigma) + z/sigma, where sigma = nu.
    - Infeasible samples where the inner term <= 0 are masked out; they do not
      contribute to the gradient. This prevents gradient distortion from
      constraint violations during early training.
    """
    # Inner argument: 1 + xi*(xi+1)*z/nu
    inner = 1.0 + xi * (xi + 1.0) * z / nu

    # Mask infeasible observations (inner <= 0): exclude from loss entirely
    feasible = inner > 0

    # Case 1: general xi (|xi| >= 1e-5)
    # l = (1 + 1/xi) * log(inner) + log(nu) - log(xi+1)
    inner_safe = inner.clamp(min=1e-8)
    log_inner = torch.log(inner_safe)
    gpd_case = (1.0 + 1.0 / xi.clamp(min=1e-8, max=10.0)) * log_inner + torch.log(nu) - torch.log(xi + 1.0)

    # Case 2: xi near 0 — exponential limit: l = log(nu) + z/nu
    exp_case = torch.log(nu) + z / nu

    # Select based on |xi|
    near_zero = xi.abs() < 1e-5
    loss_per_sample = torch.where(near_zero, exp_case, gpd_case)

    # Zero out infeasible samples
    loss_per_sample = torch.where(feasible, loss_per_sample, torch.zeros_like(loss_per_sample))

    n_feasible = feasible.float().sum().clamp(min=1.0)

    if reduction == "none":
        return loss_per_sample
    elif reduction == "sum":
        return loss_per_sample.sum()
    else:  # mean
        return loss_per_sample.sum() / n_feasible


def sigma_from_nu_xi(nu: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Recover sigma from orthogonal parameterisation: sigma = nu / (xi + 1).

    Parameters
    ----------
    nu:
        Orthogonal scale parameter (> 0).
    xi:
        Shape parameter (> -1).

    Returns
    -------
    torch.Tensor
        Scale parameter sigma.
    """
    return nu / (xi + 1.0)


def sigma_from_nu_xi_numpy(nu: ArrayLike, xi: ArrayLike) -> np.ndarray:
    """NumPy version of sigma recovery."""
    nu = np.asarray(nu, dtype=float)
    xi = np.asarray(xi, dtype=float)
    return nu / (xi + 1.0)


# ---------------------------------------------------------------------------
# Analytical validation utilities (for testing)
# ---------------------------------------------------------------------------


def scipy_gpd_quantile(p: float, xi: float, sigma: float, loc: float = 0.0) -> float:
    """Quantile using scipy.stats.genpareto for validation.

    scipy's genpareto uses c = xi (shape), scale = sigma, loc = loc.
    """
    return float(stats.genpareto.ppf(p, c=xi, scale=sigma, loc=loc))


def ogpd_loss_analytical(z: float, nu: float, xi: float) -> float:
    """Analytical evaluation of the orthogonal GPD loss for a single observation.

    Used in unit tests to verify the tensor implementation.

    Parameters
    ----------
    z:
        Single excess value > 0.
    nu:
        Orthogonal scale parameter > 0.
    xi:
        Shape parameter.

    Returns
    -------
    float
        Loss value.
    """
    if abs(xi) < 1e-5:
        return float(np.log(nu) + z / nu)
    inner = 1.0 + xi * (xi + 1.0) * z / nu
    if inner <= 0:
        raise ValueError(f"Infeasible: inner = {inner:.4f} <= 0")
    return float((1.0 + 1.0 / xi) * np.log(inner) + np.log(nu) - np.log(xi + 1.0))
