"""
Exceedance probability curves for portfolio-level tail analysis.

The occurrence exceedance probability (OEP) curve gives P(any single
risk loss > x) as a function of x. It is derived from the quantile
predictions across the portfolio and is used for:

- Reinsurance pricing: determining XL attachment points
- Catastrophe model benchmarking
- Management reporting of tail risk

The aggregate exceedance probability (AEP) curve would require simulation
of correlated losses across the portfolio, which is beyond scope here.
This module focuses on the per-occurrence OEP, which can be derived
analytically from the marginal loss distributions.

For independent risks, P(max loss > x) = 1 - product(1 - P(Y_i > x)),
but this is only meaningful for a small number of large independent risks.
For a large heterogeneous portfolio, the mean exceedance curve is more
useful as a risk profile summary.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._types import ExceedanceCurve

__all__ = [
    "exceedance_curve",
    "oep_curve",
]


def exceedance_curve(
    model: "QuantileGBM",  # noqa: F821
    X: pl.DataFrame,
    thresholds: list[float] | None = None,
    n_thresholds: int = 100,
) -> pl.DataFrame:
    """
    Compute the mean exceedance probability curve for a portfolio.

    For each threshold x in the grid, computes the average P(Y_i > x) across
    all risks in X. This gives the expected exceedance probability for a
    randomly selected risk from the portfolio.

    Parameters
    ----------
    model:
        A fitted QuantileGBM.
    X:
        Feature matrix representing the portfolio.
    thresholds:
        Specific threshold values at which to evaluate P(Y > x).
        If None, a grid from 0 to the 99th quantile of predictions is used.
    n_thresholds:
        Number of threshold points when thresholds is None. Default 100.

    Returns
    -------
    Polars DataFrame with columns:
        - threshold: the loss threshold x
        - exceedance_prob: average P(Y > x) across the portfolio
        - n_risks: number of risks (constant column for traceability)

    Notes
    -----
    The exceedance probability at x=0 is always 1.0 if all risks have
    positive expected loss. For zero-inflated portfolios (many risks with
    zero claim probability), the curve will start below 1.0.
    """
    preds = model.predict(X)
    quantiles = model.spec.quantiles
    col_names = model.spec.column_names
    pred_matrix = np.stack([preds[c].to_numpy() for c in col_names], axis=1)
    qs = np.array(quantiles)

    n_risks = pred_matrix.shape[0]

    if thresholds is None:
        # Build grid from 0 to slightly above the portfolio 99th quantile
        max_q_col = col_names[-1]
        max_val = float(preds[max_q_col].max())
        thresholds = list(np.linspace(0, max_val * 1.05, n_thresholds))

    x_arr = np.array(thresholds)
    exceedance = np.zeros(len(x_arr))

    for i in range(n_risks):
        row = pred_matrix[i]
        # S(x) = 1 - alpha(x), interpolated from quantile predictions
        alpha_at_x = np.interp(x_arr, row, qs, left=0.0, right=1.0)
        surv = 1.0 - alpha_at_x
        exceedance += surv

    exceedance /= n_risks  # average across portfolio

    return pl.DataFrame(
        {
            "threshold": list(thresholds),
            "exceedance_prob": exceedance.tolist(),
            "n_risks": [n_risks] * len(thresholds),
        }
    )


def oep_curve(
    model: "QuantileGBM",  # noqa: F821
    X: pl.DataFrame,
    thresholds: list[float] | None = None,
    n_thresholds: int = 100,
    independence_assumption: bool = False,
) -> ExceedanceCurve:
    """
    Compute the occurrence exceedance probability (OEP) curve.

    The OEP curve is standard output for catastrophe models and XL
    reinsurance pricing. It gives P(maximum single-risk loss > x) for
    a portfolio.

    Two methods:
    1. Mean exceedance (default): returns the average P(Y_i > x) per risk.
       This is appropriate for large homogeneous portfolios and is stable.
    2. True OEP under independence: P(max > x) = 1 - product(1 - P(Y_i > x)).
       Mathematically correct for independent risks but numerically unstable
       for large portfolios (probability product underflows). Only use for
       portfolios of <1000 risks.

    Parameters
    ----------
    model:
        A fitted QuantileGBM.
    X:
        Feature matrix for the portfolio.
    thresholds:
        Loss threshold grid.
    n_thresholds:
        Number of threshold points when thresholds is None.
    independence_assumption:
        If True, compute the true OEP under independence:
        P(max > x) = 1 - prod(P(Y_i <= x)).
        If False (default), use mean exceedance probability.

    Returns
    -------
    ExceedanceCurve object with .as_dataframe() method.
    """
    preds = model.predict(X)
    quantiles = model.spec.quantiles
    col_names = model.spec.column_names
    pred_matrix = np.stack([preds[c].to_numpy() for c in col_names], axis=1)
    qs = np.array(quantiles)

    n_risks = pred_matrix.shape[0]

    if thresholds is None:
        max_q_col = col_names[-1]
        max_val = float(preds[max_q_col].max())
        thresholds = list(np.linspace(0, max_val * 1.05, n_thresholds))

    x_arr = np.array(thresholds)

    if independence_assumption:
        # True OEP: P(max > x) = 1 - prod_i(P(Y_i <= x))
        # Compute log-sum for numerical stability
        log_cdf_sum = np.zeros(len(x_arr))
        for i in range(n_risks):
            row = pred_matrix[i]
            alpha_at_x = np.interp(x_arr, row, qs, left=0.0, right=1.0)
            cdf = alpha_at_x.clip(1e-15, 1.0)
            log_cdf_sum += np.log(cdf)
        probs = (1.0 - np.exp(log_cdf_sum)).tolist()
    else:
        # Mean exceedance
        exceedance = np.zeros(len(x_arr))
        for i in range(n_risks):
            row = pred_matrix[i]
            alpha_at_x = np.interp(x_arr, row, qs, left=0.0, right=1.0)
            exceedance += 1.0 - alpha_at_x
        probs = (exceedance / n_risks).tolist()

    return ExceedanceCurve(
        thresholds=list(thresholds),
        probabilities=probs,
        n_risks=n_risks,
    )
