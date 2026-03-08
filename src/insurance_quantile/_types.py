"""
Type definitions and dataclasses for insurance-quantile.

These types form the vocabulary the library uses when returning results.
They are designed to read naturally to a pricing actuary rather than
a machine learning engineer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

__all__ = [
    "QuantileSpec",
    "TailModel",
    "TVaRResult",
    "ExceedanceCurve",
]


@dataclass(frozen=True)
class QuantileSpec:
    """
    Specification for a set of quantiles to model.

    Parameters
    ----------
    quantiles:
        Probability levels, each in (0, 1). For example, [0.5, 0.75, 0.9, 0.95, 0.99].
        Must be strictly increasing.
    mode:
        'quantile' uses CatBoost MultiQuantile loss (all quantiles in one model).
        'expectile' fits a separate CatBoost model per alpha. Expectile regression
        is both coherent and elicitable — a unique property that makes it preferable
        to quantile regression on heavy-tailed lines such as motor bodily injury.

    Raises
    ------
    ValueError
        If quantiles are not strictly increasing or not in (0, 1).
    """

    quantiles: list[float]
    mode: Literal["quantile", "expectile"] = "quantile"

    def __post_init__(self) -> None:
        if not self.quantiles:
            raise ValueError("quantiles must be a non-empty list")
        for q in self.quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(f"Each quantile must be in (0, 1), got {q}")
        for a, b in zip(self.quantiles, self.quantiles[1:]):
            if a >= b:
                raise ValueError(
                    f"quantiles must be strictly increasing, got {a} >= {b}"
                )

    @property
    def column_names(self) -> list[str]:
        """Column names used in predict() output, e.g. ['q_0.5', 'q_0.9']."""
        return [f"q_{q}" for q in self.quantiles]


@dataclass
class TailModel:
    """
    Container for a fitted QuantileGBM model and its specification.

    Returned by QuantileGBM.fit(). Holds enough metadata to reproduce
    predictions and document how the model was built.

    Attributes
    ----------
    spec:
        The QuantileSpec used to fit this model.
    n_features:
        Number of input features.
    feature_names:
        Feature column names as seen during fit.
    n_training_rows:
        Number of rows used in training (after any exposure filtering).
    catboost_params:
        The CatBoost parameters used (copy, for audit purposes).
    fix_crossing:
        Whether isotonic regression post-processing was applied.
    """

    spec: QuantileSpec
    n_features: int
    feature_names: list[str]
    n_training_rows: int
    catboost_params: dict = field(default_factory=dict)
    fix_crossing: bool = True


@dataclass
class TVaRResult:
    """
    Per-risk Tail Value at Risk results.

    Attributes
    ----------
    alpha:
        The probability level used. TVaR_alpha = E[Y | Y > VaR_alpha].
    values:
        Per-risk TVaR estimates as a Polars Series.
    var_values:
        The corresponding VaR (quantile) at alpha for each risk.
        Stored alongside TVaR for audit; TVaR should always exceed VaR.
    method:
        How TVaR was approximated. 'grid_mean' means we took the mean of
        quantile predictions at levels above alpha.
    """

    alpha: float
    values: pl.Series
    var_values: pl.Series
    method: str = "grid_mean"

    @property
    def loading_over_var(self) -> pl.Series:
        """Additive loading: TVaR - VaR per risk."""
        return self.values - self.var_values


@dataclass
class ExceedanceCurve:
    """
    Portfolio occurrence exceedance probability (OEP) curve.

    Attributes
    ----------
    thresholds:
        Loss thresholds at which exceedance probability is evaluated.
    probabilities:
        P(loss > threshold) at each threshold, averaged across the portfolio.
    n_risks:
        Number of risks in the portfolio.
    """

    thresholds: list[float]
    probabilities: list[float]
    n_risks: int

    def as_dataframe(self) -> pl.DataFrame:
        """Return the curve as a two-column Polars DataFrame."""
        return pl.DataFrame(
            {
                "threshold": self.thresholds,
                "exceedance_prob": self.probabilities,
            }
        )
