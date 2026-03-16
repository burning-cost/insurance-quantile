"""
Tail Value at Risk (TVaR) calculations for insurance portfolios.

TVaR (also called Conditional Tail Expectation, CTE, or Expected Shortfall
in the banking literature) is defined as:

    TVaR_alpha(Y) = E[Y | Y > VaR_alpha(Y)]

It is a coherent risk measure — it satisfies subadditivity, which means
TVaR(portfolio) <= sum(TVaR(individual risks)). This is the key property
that makes it suitable for capital allocation and aggregate excess of loss
pricing, unlike VaR which is not subadditive.

The approximation used here integrates the quantile function numerically
using the trapezoidal rule:

    TVaR_alpha ≈ (1 / (1 - alpha)) * integral_{alpha}^{1} Q(u) du
               ≈ trapz(Q(u), u) / (1 - alpha)
                 where u is the grid of quantile levels above alpha

We anchor the integral at Q(alpha) when the model includes alpha as one of its
fitted quantile levels. When alpha is not a fitted level, we use Q(first_above)
as a conservative flat extrapolation for the left boundary. Both choices
guarantee TVaR >= VaR since the integrand Q(u) >= Q(alpha) for all u > alpha
when quantile predictions are monotone.

This is more accurate than a naive mean of Q(u) at available levels, which
gives equal weight to unevenly spaced quantile levels. The trapezoidal rule
weights by interval width: with levels [0.95, 0.99] at alpha=0.9, the
interval (0.9, 0.95) has width 5/10 and (0.95, 0.99) has width 4/10, which
is handled correctly.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# numpy<2.0 compat: trapezoid was added in 2.0, trapz deprecated in 2.0
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

from ._types import TVaRResult

__all__ = [
    "per_risk_tvar",
    "portfolio_tvar",
]


def per_risk_tvar(
    model: "QuantileGBM",  # noqa: F821  (forward reference)
    X: pl.DataFrame,
    alpha: float,
) -> TVaRResult:
    """
    Compute per-risk Tail Value at Risk (TVaR) at confidence level alpha.

    TVaR_alpha(i) = E[Y_i | Y_i > VaR_alpha(Y_i)]

    This is approximated using trapezoidal integration over the model's
    quantile predictions. When the model includes alpha as a fitted quantile
    level, the integral starts at Q(alpha) — this is both the most accurate
    left boundary and guarantees TVaR >= VaR. When alpha is not a fitted
    level, the first quantile above alpha is used as a flat extrapolation.

    Accuracy improves with the number of quantile levels above alpha that
    were fitted in the model — for best results, include levels 0.9, 0.95,
    0.975, 0.99, 0.995 when fitting.

    Parameters
    ----------
    model:
        A fitted QuantileGBM instance.
    X:
        Feature matrix for which to compute TVaR.
    alpha:
        Confidence level. TVaR_0.95 is the expected loss given that the loss
        exceeds its 95th percentile.

    Returns
    -------
    TVaRResult containing per-risk TVaR and VaR estimates.

    Raises
    ------
    ValueError
        If alpha is not in (0, 1) or if the model has no quantile levels
        strictly above alpha.
    RuntimeError
        If the model is not fitted.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    above = [q for q in model.spec.quantiles if q > alpha]
    if not above:
        raise ValueError(
            f"Model has no quantile levels above alpha={alpha}. "
            f"Model quantiles: {model.spec.quantiles}. "
            "Add higher quantile levels (e.g. 0.99) when constructing QuantileGBM."
        )

    preds = model.predict(X)

    # --- Determine left boundary value at alpha ---
    # If the model has a quantile level exactly at alpha, use Q(alpha) as the
    # left boundary. This is both accurate and guarantees TVaR >= VaR since
    # the integrand Q(u) >= Q(alpha) for u > alpha (quantile monotonicity).
    # If alpha is not a fitted level, fall back to flat extrapolation using
    # Q(first_above), which is a conservative overestimate for the interval
    # [alpha, first_above].
    at_alpha = [q for q in model.spec.quantiles if q == alpha]
    if at_alpha:
        boundary_col = f"q_{at_alpha[0]}"
        boundary_vals = preds[boundary_col].to_numpy()
    else:
        # Flat extrapolation: use the lowest tail quantile as boundary value.
        # This slightly overestimates TVaR (since Q(u) >= boundary for u > alpha),
        # but guarantees TVaR >= boundary >= VaR (when boundary >= VaR).
        boundary_vals = preds[f"q_{above[0]}"].to_numpy()

    # --- Build integration grid ---
    tail_cols = [f"q_{q}" for q in above]
    tail_matrix = np.stack([preds[c].to_numpy() for c in tail_cols], axis=1)
    above_arr = np.array(above)

    # Prepend alpha as the left boundary, then the tail levels
    levels = np.concatenate([[alpha], above_arr])
    # shape: (n_risks, n_levels)
    boundary_col_matrix = boundary_vals.reshape(-1, 1)
    values_with_boundary = np.concatenate([boundary_col_matrix, tail_matrix], axis=1)

    # Trapezoidal integration: integral_{alpha}^{max_quantile} Q(u) du
    # Divide by (1 - alpha) to get the TVaR estimate.
    # Note: we are missing the tail beyond max_quantile (e.g. [0.99, 1.0]).
    # For this reason, include high quantile levels (0.995, 0.999) when
    # precision in the extreme tail matters.
    integral = _trapezoid(values_with_boundary, levels, axis=1)  # shape: (n_risks,)
    tvar_vals = integral / (1.0 - alpha)

    # Enforce TVaR >= VaR (numerical safety: isotonic regression may allow
    # tiny violations; clip to be safe)
    at_or_below = [q for q in model.spec.quantiles if q <= alpha]
    if at_or_below:
        var_col = f"q_{at_or_below[-1]}"
        var_vals = preds[var_col].to_numpy()
    else:
        var_col = f"q_{model.spec.quantiles[0]}"
        var_vals = preds[var_col].to_numpy()

    tvar_vals = np.maximum(tvar_vals, var_vals)

    return TVaRResult(
        alpha=alpha,
        values=pl.Series("tvar", tvar_vals),
        var_values=pl.Series("var", var_vals),
        method="trapezoidal",
    )


def portfolio_tvar(
    model: "QuantileGBM",  # noqa: F821
    X: pl.DataFrame,
    alpha: float,
    aggregate_method: str = "mean",
) -> float:
    """
    Compute a single portfolio-level TVaR estimate.

    This aggregates per-risk TVaR values into a single number. Note that
    this is NOT the TVaR of the portfolio aggregate loss distribution — that
    would require simulation. Instead, this is the average (or sum) of per-risk
    TVaRs, useful as a summary statistic for portfolio tail risk.

    For the true portfolio aggregate TVaR, you need a copula model or
    simulation framework. This function is suitable for:
    - Comparing tail risk across rating factors or segments
    - Generating per-risk large loss loadings summed to a portfolio total
    - Reporting average tail risk as a monitoring metric

    Parameters
    ----------
    model:
        A fitted QuantileGBM instance.
    X:
        Feature matrix for the portfolio.
    alpha:
        Confidence level.
    aggregate_method:
        'mean' returns average TVaR across risks.
        'sum' returns total (sum of per-risk TVaRs).

    Returns
    -------
    Single float: the portfolio-level TVaR aggregate.
    """
    result = per_risk_tvar(model, X, alpha)
    vals = result.values.to_numpy()
    if aggregate_method == "mean":
        return float(vals.mean())
    elif aggregate_method == "sum":
        return float(vals.sum())
    else:
        raise ValueError(
            f"aggregate_method must be 'mean' or 'sum', got '{aggregate_method}'"
        )
