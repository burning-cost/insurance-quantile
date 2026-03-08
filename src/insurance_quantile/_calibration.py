"""
Calibration diagnostics for quantile regression models.

A quantile model is calibrated if the stated probability level matches
the observed coverage fraction: your q_0.9 prediction should be exceeded
by roughly 10% of observations. This module provides the checks.

Pinball loss (also called the check loss or tick loss) is the standard
scoring rule for quantile regression. It is strictly proper for quantiles,
meaning it is minimised in expectation by the true quantile function.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = [
    "pinball_loss",
    "coverage_check",
    "quantile_calibration_plot",
]


def pinball_loss(
    y_true: pl.Series,
    y_pred: pl.Series,
    alpha: float,
) -> float:
    """
    Compute the pinball (check) loss for a single quantile level.

    The pinball loss at level alpha is:

        L(y, q) = alpha * max(y - q, 0) + (1 - alpha) * max(q - y, 0)

    It is also written as:

        L(y, q) = (alpha - I(y < q)) * (y - q)

    This is strictly proper for quantile regression: it is minimised in
    expectation by the true alpha-quantile of the conditional distribution.
    Use it to compare models at the same alpha level; do not compare across
    alpha levels without normalisation.

    Parameters
    ----------
    y_true:
        Observed values.
    y_pred:
        Predicted quantile at level alpha.
    alpha:
        Quantile level, in (0, 1).

    Returns
    -------
    Mean pinball loss across all observations.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    y = y_true.to_numpy().astype(np.float64)
    q = y_pred.to_numpy().astype(np.float64)
    residual = y - q
    loss = np.where(residual >= 0, alpha * residual, (alpha - 1.0) * residual)
    return float(loss.mean())


def coverage_check(
    y_true: pl.Series,
    predictions: pl.DataFrame,
    quantiles: list[float],
) -> pl.DataFrame:
    """
    Compute observed coverage fraction for each quantile level.

    For a well-calibrated quantile model, the fraction of observations
    where y <= q_alpha should be approximately alpha. Systematic deviations
    indicate model miscalibration — either in the tails (common for heavy-tailed
    lines) or in the body of the distribution.

    Parameters
    ----------
    y_true:
        Observed values.
    predictions:
        DataFrame of quantile predictions, e.g. from QuantileGBM.predict().
        Columns should be named q_0.5, q_0.9, etc.
    quantiles:
        The quantile levels corresponding to columns in predictions.
        Must be in the same order as predictions.columns.

    Returns
    -------
    Polars DataFrame with columns:
        - quantile: the probability level
        - column: column name in predictions
        - expected_coverage: equal to quantile (what a calibrated model gives)
        - observed_coverage: fraction of y_true <= predicted quantile
        - coverage_error: observed_coverage - expected_coverage
    """
    y = y_true.to_numpy().astype(np.float64)
    col_names = [f"q_{q}" for q in quantiles]

    rows_q: list[float] = []
    rows_col: list[str] = []
    rows_expected: list[float] = []
    rows_observed: list[float] = []
    rows_error: list[float] = []

    for q, col in zip(quantiles, col_names):
        if col not in predictions.columns:
            raise ValueError(
                f"Expected column '{col}' not found in predictions. "
                f"Available: {predictions.columns}"
            )
        q_pred = predictions[col].to_numpy().astype(np.float64)
        observed = float(np.mean(y <= q_pred))
        rows_q.append(q)
        rows_col.append(col)
        rows_expected.append(q)
        rows_observed.append(observed)
        rows_error.append(observed - q)

    return pl.DataFrame(
        {
            "quantile": rows_q,
            "column": rows_col,
            "expected_coverage": rows_expected,
            "observed_coverage": rows_observed,
            "coverage_error": rows_error,
        }
    )


def quantile_calibration_plot(
    y_true: pl.Series,
    predictions: pl.DataFrame,
    quantiles: list[float],
    title: str = "Quantile calibration",
) -> None:
    """
    Plot observed vs expected coverage for each quantile level.

    A calibrated model lies on the diagonal. Points above the diagonal
    mean the model is conservative (predicts too high); points below
    mean it is optimistic (under-predicts tail risk).

    Parameters
    ----------
    y_true:
        Observed values.
    predictions:
        DataFrame of quantile predictions from QuantileGBM.predict().
    quantiles:
        Quantile levels corresponding to columns in predictions.
    title:
        Plot title.

    Notes
    -----
    Requires matplotlib. Install via: pip install insurance-quantile[plot]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plots. "
            "Install via: pip install insurance-quantile[plot]"
        )

    calib = coverage_check(y_true, predictions, quantiles)
    expected = calib["expected_coverage"].to_list()
    observed = calib["observed_coverage"].to_list()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.scatter(expected, observed, zorder=5, color="steelblue", s=60)
    for e, o, q in zip(expected, observed, quantiles):
        ax.annotate(f"q_{q}", (e, o), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Expected coverage (quantile level)")
    ax.set_ylabel("Observed coverage (fraction of y ≤ predicted)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
