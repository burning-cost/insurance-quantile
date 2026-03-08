"""
Tail Value at Risk (TVaR) calculations for insurance portfolios.

TVaR (also called Conditional Tail Expectation, CTE, or Expected Shortfall
in the banking literature) is defined as:

    TVaR_alpha(Y) = E[Y | Y > VaR_alpha(Y)]

It is a coherent risk measure — it satisfies subadditivity, which means
TVaR(portfolio) <= sum(TVaR(individual risks)). This is the key property
that makes it suitable for capital allocation and aggregate excess of loss
pricing, unlike VaR which is not subadditive.

The approximation used here integrates the quantile function numerically:

    TVaR_alpha ≈ (1 / (1 - alpha)) * integral_{alpha}^{1} Q(u) du
               ≈ mean(Q(u) for u in grid above alpha)

where Q(u) is the predicted quantile at level u.
"""

from __future__ import annotations

import numpy as np
import polars as pl

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

    This is approximated by taking the mean of the model's quantile predictions
    at levels strictly above alpha. The accuracy improves with the number of
    quantile levels above alpha that were fitted in the model — for best results,
    include levels 0.9, 0.95, 0.975, 0.99, 0.995 when fitting.

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
    tail_cols = [f"q_{q}" for q in above]
    tail_matrix = np.stack([preds[c].to_numpy() for c in tail_cols], axis=1)
    tvar_vals = tail_matrix.mean(axis=1)

    # VaR at alpha: use closest quantile at or below alpha
    at_or_below = [q for q in model.spec.quantiles if q <= alpha]
    if at_or_below:
        var_col = f"q_{at_or_below[-1]}"
        var_vals = preds[var_col].to_numpy()
    else:
        # No quantile at or below alpha; use the lowest available as a floor
        var_col = f"q_{model.spec.quantiles[0]}"
        var_vals = preds[var_col].to_numpy()

    return TVaRResult(
        alpha=alpha,
        values=pl.Series("tvar", tvar_vals),
        var_values=pl.Series("var", var_vals),
        method="grid_mean",
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
