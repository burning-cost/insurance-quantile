"""
Large loss loading and increased limits factors for insurance pricing.

Large loss loading is the actuarial adjustment added to the burning cost
rate to cover catastrophic individual claims. Standard pricing models
(Tweedie GLM/GBM) estimate the mean of the full loss distribution, which
may understate the expected cost for risks with heavy-tailed severity.

The loading approach here:

    loading_i = TVaR_alpha(i) - E[Y_i]

where TVaR comes from a QuantileGBM and E[Y] from a separate Tweedie model.
The loading is additive: it represents the amount by which the rate should
exceed the pure burning cost to cover large losses with adequate confidence.

Increased Limits Factors (ILF) are the rating mechanism for excess layers.
ILF(L1, L2) = E[min(Y, L2)] / E[min(Y, L1)] is the factor by which the
rate at basic limit L1 must be multiplied to price coverage up to L2.
We estimate E[min(Y, L)] by integrating the exceedance curve:

    E[min(Y, L)] = integral_0^L P(Y > x) dx

which is estimated numerically from the quantile predictions.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# numpy<2.0 compat: trapezoid was added in 2.0, trapz deprecated in 2.0
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "large_loss_loading",
    "ilf",
]


def large_loss_loading(
    model_mean: "Any",  # noqa: F821 — Tweedie/GBM model with .predict(X) -> array-like
    model_quantile: "QuantileGBM",  # noqa: F821
    X: pl.DataFrame,
    alpha: float = 0.95,
) -> pl.Series:
    """
    Compute per-risk additive large loss loading.

    The large loss loading captures the excess of the tail expectation
    (TVaR) over the mean predicted loss. It is an additive adjustment
    to the burning cost premium:

        loading_i = TVaR_alpha(i) - E[Y_i]

    Parameters
    ----------
    model_mean:
        Any model with a predict method that returns mean loss estimates.
        Accepts models that take Polars DataFrames, numpy arrays, or both.
        For raw CatBoostRegressor or sklearn estimators that do not accept
        Polars DataFrames, the feature matrix is automatically converted to
        a numpy array before prediction.
    model_quantile:
        A fitted QuantileGBM, used to derive TVaR_alpha per risk.
    X:
        Feature matrix as a Polars DataFrame.
    alpha:
        TVaR confidence level. 0.95 is standard for personal lines.
        Use 0.99 for commercial lines where aggregate stop-loss treaties
        are priced at higher confidence levels.

    Returns
    -------
    Polars Series of per-risk large loss loadings. Values may be negative
    if the TVaR estimate is below the mean prediction (unusual but possible
    for risks with very light tails or when using a low alpha).

    Notes
    -----
    The model_mean predictions should be on the same scale as y_true — if
    model_mean predicts claim frequency separately and model_quantile was
    fitted on severity, the scales will differ. Ensure consistency in how
    both models were trained.
    """
    from ._tvar import per_risk_tvar

    tvar_result = per_risk_tvar(model_quantile, X, alpha)
    tvar_vals = tvar_result.values.to_numpy()

    # Try passing X as-is first; if that fails (e.g. raw CatBoost/sklearn
    # models that do not accept Polars DataFrames), fall back to numpy.
    X_np = X.to_numpy()
    try:
        mean_preds = model_mean.predict(X)
        if isinstance(mean_preds, pl.DataFrame):
            if mean_preds.width != 1:
                raise ValueError(
                    "model_mean.predict(X) returned a multi-column DataFrame. "
                    "Expected a single-column DataFrame or Series."
                )
            mean_vals = mean_preds.to_series().to_numpy().astype(np.float64)
        elif isinstance(mean_preds, pl.Series):
            mean_vals = mean_preds.to_numpy().astype(np.float64)
        else:
            # numpy array or other array-like
            mean_vals = np.asarray(mean_preds, dtype=np.float64)
    except (TypeError, ValueError, AttributeError):
        # Model does not accept Polars input — retry with numpy array
        mean_preds = model_mean.predict(X_np)
        mean_vals = np.asarray(mean_preds, dtype=np.float64)

    loading = tvar_vals - mean_vals
    return pl.Series("large_loss_loading", loading)


def ilf(
    model: "QuantileGBM",  # noqa: F821
    X: pl.DataFrame,
    basic_limit: float,
    higher_limit: float,
    n_integration_points: int = 200,
) -> pl.Series:
    """
    Compute per-risk Increased Limits Factors (ILF).

    ILF(L1, L2) = E[min(Y, L2)] / E[min(Y, L1)]

    This is the factor by which the rate at basic limit L1 should be
    multiplied to price coverage up to limit L2. It is estimated by
    numerically integrating the exceedance curve derived from the
    quantile predictions:

        E[min(Y, L)] = integral_0^L S(x) dx
                     = integral_0^L P(Y > x) dx

    where S(x) is the survival function, approximated from quantile predictions.

    Parameters
    ----------
    model:
        A fitted QuantileGBM. For accurate ILFs, the model should include
        high quantile levels (0.99, 0.995 or higher) so the tail beyond
        the basic limit is well-estimated.
    X:
        Feature matrix.
    basic_limit:
        The lower (basic) policy limit, e.g. 100_000 for £100k.
    higher_limit:
        The upper limit for which to compute the ILF, e.g. 500_000 for £500k.
    n_integration_points:
        Number of quadrature points for numerical integration. 200 is
        sufficient for smooth exceedance curves; increase if you observe
        instability in ILF values.

    Returns
    -------
    Polars Series of per-risk ILF values. A value of 1.0 means no extra
    loading; values above 1.0 indicate the higher limit costs more.

    Raises
    ------
    ValueError
        If basic_limit >= higher_limit or either limit is non-positive.
    """
    if basic_limit <= 0 or higher_limit <= 0:
        raise ValueError("Both limits must be positive.")
    if basic_limit >= higher_limit:
        raise ValueError(
            f"basic_limit ({basic_limit}) must be less than higher_limit ({higher_limit})"
        )

    preds = model.predict(X)
    quantiles = model.spec.quantiles
    col_names = model.spec.column_names
    pred_matrix = np.stack([preds[c].to_numpy() for c in col_names], axis=1)

    n_risks = pred_matrix.shape[0]
    ilf_vals = np.empty(n_risks)

    # Integration grids
    x_basic = np.linspace(0, basic_limit, n_integration_points + 1)
    x_higher = np.linspace(0, higher_limit, n_integration_points + 1)

    qs = np.array(quantiles)  # shape (n_q,)

    for i in range(n_risks):
        row = pred_matrix[i]  # quantile predictions for this risk, shape (n_q,)
        # Estimate survival function S(x) = P(Y > x) at a given x
        # by interpolating from the quantile predictions.
        # Q(alpha) = x => S(x) = P(Y > x) = 1 - alpha
        # So S(x) = 1 - alpha(x), where alpha(x) is found by inverting Q.

        def survival(x_vals: np.ndarray) -> np.ndarray:
            # For each x, find the interpolated quantile level alpha such that Q(alpha) = x.
            # S(x) = 1 - alpha
            # np.interp requires increasing xp; quantile predictions should be non-decreasing.
            # Clip to [0, 1] to handle x outside model range.
            alpha_at_x = np.interp(x_vals, row, qs, left=0.0, right=1.0)
            return 1.0 - alpha_at_x

        # Integrate S(x) from 0 to L using the trapezoidal rule
        e_min_basic = _trapezoid(survival(x_basic), x_basic)
        e_min_higher = _trapezoid(survival(x_higher), x_higher)

        if e_min_basic <= 0:
            ilf_vals[i] = 1.0
        else:
            ilf_vals[i] = e_min_higher / e_min_basic

    return pl.Series("ilf", ilf_vals)
