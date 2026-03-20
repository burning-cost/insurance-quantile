"""
TwoPartQuantilePremium: frequency-severity quantile premium decomposition.

The two-part quantile premium principle (QPP) prices zero-inflated insurance
risks at an explicit aggregate confidence level. It solves a real problem:
fitting a QuantileGBM directly on a zero-inflated loss distribution produces
trivial (zero) quantile estimates for low-frequency risks when tau < p_i.

The approach — from Heras, Moreno & Vilar-Zanon (2018) / Laporta et al. (2024)
and the NAAJ 2025 ML extension — maps the desired aggregate quantile tau to an
adjusted severity quantile tau_i via:

    tau_i = (tau - p_i) / (1 - p_i)

where p_i = Pr(N_i = 0 | x_i) is the no-claim probability. The premium is
then a blend between the conditional severity quantile and the pure premium:

    P_i = gamma * Q~_{tau_i}(x_i) + (1 - gamma) * E[S_i | x_i]

This gives a formal, risk-specific safety loading derived from an explicit
confidence level — not an ad hoc percentage applied uniformly across the book.

UK motor OD example: p_i = 0.80, tau = 0.90 -> tau_i = 0.50 (severity median).
The severity model is asked for a well-estimated interior quantile rather than
a 90th percentile that would be zero for 80% of the distribution.

Sources:
- Heras et al. 2018: https://doi.org/10.1080/03461238.2018.1452786
- Laporta et al. 2024: https://doi.org/10.1017/S1748499523000106
- NAAJ 2025 (ML extension): https://doi.org/10.1080/10920277.2025.2503744
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl

from ._types import TwoPartResult

__all__ = ["TwoPartQuantilePremium"]

# numpy<2.0 compat
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _interpolate_severity_quantile(
    q_matrix: np.ndarray,
    q_levels: np.ndarray,
    tau_i: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Vectorised piecewise-linear interpolation of the quantile function.

    For each valid policy i, finds the two adjacent quantile levels that
    bracket tau_i[i] and linearly interpolates the corresponding quantile
    predictions.

    Parameters
    ----------
    q_matrix:
        Shape (n, n_q). Row i contains QuantileGBM predictions at each
        quantile level for policy i.
    q_levels:
        Shape (n_q,). The quantile levels, strictly increasing.
    tau_i:
        Shape (n,). Per-policy adjusted severity quantile; NaN for fallback
        policies (where valid_mask is False).
    valid_mask:
        Boolean array of shape (n,). True where tau_i is valid (in (0, 1)).

    Returns
    -------
    result:
        Shape (n,). Interpolated severity quantile for valid policies, NaN
        elsewhere.
    extrapolation_fraction:
        Fraction of valid policies where tau_i exceeds max(q_levels). These
        receive flat extrapolation (held at the highest quantile prediction).
    """
    n = len(tau_i)
    n_q = len(q_levels)
    result = np.full(n, np.nan)

    valid_tau = tau_i[valid_mask]          # shape (n_valid,)
    if len(valid_tau) == 0:
        return result, 0.0

    q_sub = q_matrix[valid_mask]           # shape (n_valid, n_q)

    # For each valid policy, find the largest index j such that q_levels[j] <= tau_i.
    # This gives the left bracket of the interpolation interval.
    # Broadcasting: (n_valid, 1) vs (1, n_q) -> (n_valid, n_q)
    above = q_levels[np.newaxis, :] <= valid_tau[:, np.newaxis]  # True where q_level <= tau_i
    j = above.sum(axis=1) - 1                                     # index of left bracket
    j = np.clip(j, 0, n_q - 2)                                   # guard both boundaries

    # Interpolation weights
    lo = q_levels[j]                       # left quantile level
    hi = q_levels[j + 1]                   # right quantile level
    # Avoid division by zero (degenerate grid, shouldn't happen with valid QuantileSpec)
    denom = hi - lo
    denom = np.where(denom > 0, denom, 1.0)
    w = np.clip((valid_tau - lo) / denom, 0.0, 1.0)

    idx = np.arange(len(valid_tau))
    result[valid_mask] = (1.0 - w) * q_sub[idx, j] + w * q_sub[idx, j + 1]

    # Extrapolation tracking: tau_i above max q_level gets flat right extrapolation.
    # The np.clip(w, 0, 1) already handles this — we just want to warn about it.
    extrapolated = valid_tau > q_levels[-1]
    extrapolation_fraction = float(extrapolated.mean()) if len(valid_tau) > 0 else 0.0

    return result, extrapolation_fraction


class TwoPartQuantilePremium:
    """
    Two-part quantile premium calculator following the Quantile Premium Principle.

    Implements the framework of Heras et al. (2018) / Laporta et al. (2024),
    as applied to ML models in the NAAJ 2025 paper
    (DOI: 10.1080/10920277.2025.2503744).

    The premium for policy i at aggregate quantile level tau is:

        P_i = gamma * Q~_{tau_i}(x_i) + (1 - gamma) * E[S_i | x_i]

    where tau_i = (tau - p_i) / (1 - p_i) is the adjusted severity quantile,
    p_i is the no-claim probability, Q~_{tau_i}(x_i) is the conditional
    severity quantile, and E[S_i | x_i] is the pure premium.

    The safety loading is formal and risk-specific:

        Loading_i = gamma * (Q~_{tau_i}(x_i) - E[S_i | x_i])

    This is not an ad hoc percentage; it is derived from the explicit
    confidence level tau and the risk's own no-claim probability.

    Note: the safety loading can be negative for low-frequency risks where
    the adjusted severity quantile tau_i is small (i.e. the required severity
    quantile is below the severity mean). This is mathematically correct —
    the quantile premium at a modest severity quantile level can fall below
    the expected loss. In practice, price floors or a minimum gamma are
    typically applied. See Section 7 of the spec for UK lines guidance.

    Parameters
    ----------
    freq_model:
        A fitted binary classifier with predict_proba(X) method. Must return
        an array of shape (n, 2) where column for class label 0 gives p_i.
        The no-claim class must have label 0. Typical choice: sklearn
        LogisticRegression(). Any sklearn-compatible estimator works.
    sev_model:
        A fitted QuantileGBM trained on non-zero claims only. The adjusted
        severity quantile tau_i is interpolated from this model's quantile
        grid — no refitting is required.
    mean_sev_model:
        Optional. A fitted model with predict(X) -> array-like of mean
        conditional severity E[S~_i | x_i] (conditional on a claim occurring).
        Typical choice: CatBoostRegressor(loss_function='Gamma') fitted on
        non-zero rows. If None, the mean severity is approximated by
        trapezoidal integration over the QuantileGBM's quantile function
        — reasonably accurate with 7+ quantile levels, but a dedicated mean
        model is preferred when available.

    Notes
    -----
    Train the frequency model on all rows with target I(N_i > 0) (claim
    indicator). Train the severity models on non-zero rows only. This is the
    standard two-part model structure.

    For UK motor OD with p_i ~ 0.80 and tau = 0.90, the adjusted severity
    quantile tau_i ~ 0.50 (severity median). The QuantileGBM should include
    quantile levels that cover the expected range of tau_i values — typically
    [0.40, 0.50, 0.60, 0.70, 0.80, 0.90] for motor OD at tau = 0.90.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from insurance_quantile import QuantileGBM, TwoPartQuantilePremium
    >>> freq = LogisticRegression(max_iter=500).fit(X_all.to_numpy(), y_freq)
    >>> sev = QuantileGBM(quantiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    >>> sev.fit(X_sev, y_sev_positive)
    >>> tpqp = TwoPartQuantilePremium(freq, sev)
    >>> result = tpqp.predict_premium(X_val, tau=0.95, gamma=0.5)
    >>> print(result.premium.describe())
    >>> print(result.n_fallback)
    """

    def __init__(
        self,
        freq_model: Any,
        sev_model: Any,       # QuantileGBM
        mean_sev_model: Any | None = None,
    ) -> None:
        self.freq_model = freq_model
        self.sev_model = sev_model
        self.mean_sev_model = mean_sev_model

    def predict_premium(
        self,
        X: pl.DataFrame,
        tau: float = 0.95,
        gamma: float = 0.5,
    ) -> TwoPartResult:
        """
        Compute per-risk loaded premiums at aggregate quantile level tau.

        Parameters
        ----------
        X:
            Feature matrix as a Polars DataFrame. All columns must be numeric.
            Must have the same columns as used to fit freq_model and sev_model.
        tau:
            Target aggregate quantile level for S_i, e.g. 0.95. Must be in
            (0, 1). For Solvency II SCR pricing use tau = 0.995.
        gamma:
            Loading factor in [0, 1]. At gamma = 0, the premium equals the
            pure premium (no loading). At gamma = 1, the premium equals the
            conditional severity quantile. Values in [0.3, 0.7] are typical
            for UK personal lines.

        Returns
        -------
        TwoPartResult with per-policy premiums, loadings, and diagnostic fields.
        See TwoPartResult for full field documentation.

        Raises
        ------
        ValueError
            If tau is not strictly in (0, 1).
            If gamma is not in [0, 1].
            If sev_model is not fitted.

        Warns
        -----
        UserWarning
            If any policies have p_i >= tau (fallback to pure premium).
            If more than 10% of valid tau_i values exceed the QuantileGBM's
            maximum quantile level (flat extrapolation applied; consider adding
            higher quantile levels to sev_model).
        """
        # ------------------------------------------------------------------
        # 1. Validate inputs
        # ------------------------------------------------------------------
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be strictly in (0, 1), got {tau}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not self.sev_model.is_fitted:
            raise RuntimeError("sev_model is not fitted. Call sev_model.fit() first.")

        n = len(X)

        # ------------------------------------------------------------------
        # 2. Frequency step: p_i = Pr(N_i = 0 | x_i)
        # ------------------------------------------------------------------
        X_np = X.to_numpy().astype(np.float64)

        # sklearn classifiers don't guarantee column 0 is class 0
        no_claim_class_idx = list(self.freq_model.classes_).index(0)
        p_i = self.freq_model.predict_proba(X_np)[:, no_claim_class_idx]

        # ------------------------------------------------------------------
        # 3. Adjusted severity quantile tau_i = (tau - p_i) / (1 - p_i)
        #    Valid only when 0 < tau_i < 1, i.e. p_i < tau (strictly).
        #    tau_i = 0 exactly (p_i = tau) is treated as fallback — see spec §5.4.
        # ------------------------------------------------------------------
        raw_tau_i = (tau - p_i) / np.clip(1.0 - p_i, 1e-12, None)
        # valid: tau_i must be strictly in (0, 1)
        valid_mask = (raw_tau_i > 0.0) & (raw_tau_i < 1.0)
        tau_i = np.where(valid_mask, raw_tau_i, np.nan)

        n_fallback = int((~valid_mask).sum())
        if n_fallback > 0:
            warnings.warn(
                f"{n_fallback} of {n} policies have p_i >= tau; fallback to pure premium "
                "with zero safety loading for these policies.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # 4. Severity quantile predictions from QuantileGBM
        # ------------------------------------------------------------------
        q_preds = self.sev_model.predict(X)  # pl.DataFrame
        q_levels = np.array(self.sev_model.spec.quantiles)
        col_names = self.sev_model.spec.column_names
        q_matrix = np.stack([q_preds[c].to_numpy() for c in col_names], axis=1)

        sev_q, extrap_frac = _interpolate_severity_quantile(
            q_matrix, q_levels, tau_i, valid_mask
        )

        if extrap_frac > 0.10:
            warnings.warn(
                f"{extrap_frac:.1%} of valid tau_i values exceed the QuantileGBM's maximum "
                f"quantile level ({q_levels[-1]:.3f}). Flat extrapolation is applied. "
                "For accuracy, add higher quantile levels to sev_model — e.g. include levels "
                "up to the 99th percentile or higher for the severity model.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # 5. Mean severity E[S~_i | x_i] (conditional on a claim)
        # ------------------------------------------------------------------
        if self.mean_sev_model is not None:
            try:
                raw_mean = self.mean_sev_model.predict(X)
            except (TypeError, AttributeError):
                raw_mean = self.mean_sev_model.predict(X_np)
            if isinstance(raw_mean, pl.DataFrame):
                mean_sev = raw_mean.to_series().to_numpy().astype(np.float64)
            elif isinstance(raw_mean, pl.Series):
                mean_sev = raw_mean.to_numpy().astype(np.float64)
            else:
                mean_sev = np.asarray(raw_mean, dtype=np.float64).ravel()
        else:
            # Approximate E[S~_i] = integral_0^1 Q_t(S~_i) dt via trapezoid rule.
            # This is exact for continuous distributions; for insurance severity
            # (log-normal, gamma) it is accurate to within a few % with 7+ quantiles.
            # A dedicated mean model (Tweedie GBM) is preferred when available.
            mean_sev = _trapezoid(q_matrix, q_levels, axis=1)

        # ------------------------------------------------------------------
        # 6. Pure premium E[S_i | x_i] = (1 - p_i) * E[S~_i | x_i]
        # ------------------------------------------------------------------
        claim_prob = 1.0 - p_i
        pure_premium = claim_prob * mean_sev

        # ------------------------------------------------------------------
        # 7. Loaded premium and safety loading
        # ------------------------------------------------------------------
        # For fallback policies: premium = pure_premium, loading = 0.
        # For valid policies: premium may be above or below pure_premium depending
        # on whether sev_q > mean_sev. When tau_i is small (low-frequency risk),
        # sev_q can be below the severity mean, giving a negative loading.
        # This is mathematically correct; apply a price floor in downstream code
        # if needed for commercial reasons.
        premium = np.where(
            valid_mask,
            gamma * sev_q + (1.0 - gamma) * pure_premium,
            pure_premium,
        )
        safety_loading = premium - pure_premium  # zero for fallback rows by construction

        # ------------------------------------------------------------------
        # 8. Pack results
        # ------------------------------------------------------------------
        # adjusted_tau: NaN for fallback policies
        adjusted_tau_series = pl.Series("adjusted_tau", tau_i)
        severity_quantile_series = pl.Series("severity_quantile", sev_q)

        return TwoPartResult(
            premium=pl.Series("premium", premium),
            pure_premium=pl.Series("pure_premium", pure_premium),
            safety_loading=pl.Series("safety_loading", safety_loading),
            no_claim_prob=pl.Series("no_claim_prob", p_i),
            adjusted_tau=adjusted_tau_series,
            severity_quantile=severity_quantile_series,
            n_fallback=n_fallback,
            tau=tau,
            gamma=gamma,
        )
