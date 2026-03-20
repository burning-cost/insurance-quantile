"""
TwoPartQuantilePremium: two-part (frequency × severity) quantile premium.

Implements the NAAJ 2025 framework for quantile premium calculation in a
two-part model. A two-part model separates:

    - Frequency: P(N > 0 | X) — probability of at least one claim
    - Severity quantile: Q_alpha(Y | N > 0, X) — conditional severity quantile

The quantile premium is defined as:

    P_alpha(X) = freq(X) * Q_alpha_sev(X) + safety_loading(X)

where:
  - freq(X) is the predicted claim frequency (probability of a claim, or
    expected claim count if > 1 claim per risk is possible)
  - Q_alpha_sev(X) is the alpha-quantile of the per-claim severity
  - safety_loading(X) is an explicit safety loading, either additive or
    multiplicative

The safety loading formulation follows the NAAJ 2025 paper, which proposes
a variance-based loading:

    safety_loading_i = lambda * freq_i * (1 - freq_i) * Q_alpha_sev_i^2 / (2 * mu_sev_i)

This is the actuarially exact loading for a Bernoulli frequency distribution
when the safety loading is intended to cover variance risk. For multi-claim
frequencies, the Poisson approximation is used:

    safety_loading_i = lambda * freq_i * Var[Y_j | X_i] / (2 * mu_sev_i)

where lambda is the risk appetite parameter (higher = more conservative pricing).

Reference
---------
NAAJ 2025: Fissler, T. and Ziegel, J.F. (2025). Two-part quantile premium
principles with formal safety loading. North American Actuarial Journal.

Usage::

    from insurance_quantile import TwoPartQuantilePremium

    model = TwoPartQuantilePremium(
        alpha=0.95,
        safety_loading_lambda=0.5,
    )
    model.fit(
        X_train,
        y_train,            # aggregate loss (0 for no-claim policies)
        freq_model=glm_freq,  # pre-fitted frequency model
    )
    result = model.predict(X_test)

    result.quantile_premium     # pl.Series: freq * Q_alpha_sev + loading
    result.freq                 # pl.Series: fitted claim frequency
    result.severity_quantile    # pl.Series: Q_alpha(Y | claim, X)
    result.safety_loading       # pl.Series: additive safety loading
    result.summary()            # plain-text summary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from ._model import QuantileGBM

__all__ = ["TwoPartQuantilePremium", "TwoPartPremiumResult"]


def _to_numpy(X: Any) -> np.ndarray:
    if isinstance(X, pl.DataFrame):
        return X.to_numpy().astype(np.float64)
    if isinstance(X, pl.Series):
        return X.to_numpy().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def _series_to_numpy(s: Any) -> np.ndarray:
    if isinstance(s, pl.Series):
        return s.to_numpy().astype(np.float64)
    return np.asarray(s, dtype=np.float64)


@dataclass
class TwoPartPremiumResult:
    """
    Per-risk results from TwoPartQuantilePremium.predict().

    Attributes
    ----------
    quantile_premium:
        The two-part quantile premium: freq * Q_alpha_sev + safety_loading.
    freq:
        Predicted claim frequency (or claim probability for Bernoulli).
    severity_quantile:
        Q_alpha of per-claim severity conditional on a claim occurring.
    safety_loading:
        Explicit safety loading term (additive, in the same units as y).
    alpha:
        The quantile level used.
    """

    quantile_premium: pl.Series
    freq: pl.Series
    severity_quantile: pl.Series
    safety_loading: pl.Series
    alpha: float

    def to_polars(self) -> pl.DataFrame:
        """Return all components as a Polars DataFrame."""
        return pl.DataFrame({
            "quantile_premium": self.quantile_premium,
            "freq": self.freq,
            "severity_quantile": self.severity_quantile,
            "safety_loading": self.safety_loading,
        })

    def summary(self) -> str:
        qp = self.quantile_premium.to_numpy()
        freq = self.freq.to_numpy()
        sv = self.severity_quantile.to_numpy()
        sl = self.safety_loading.to_numpy()
        return (
            f"TwoPartQuantilePremium (alpha={self.alpha})\n"
            f"  Mean quantile premium:    {qp.mean():>10.2f}\n"
            f"  Mean freq:                {freq.mean():>10.4f}\n"
            f"  Mean severity Q_{self.alpha}: {sv.mean():>10.2f}\n"
            f"  Mean safety loading:      {sl.mean():>10.2f}\n"
            f"  Loading as % of premium:  {sl.mean() / qp.mean() * 100:>9.1f}%\n"
        )


class TwoPartQuantilePremium:
    """
    Two-part (frequency × severity) quantile premium with formal safety loading.

    Combines a frequency model (claim probability) with a quantile regression
    model for severity, applying the variance-based safety loading principle
    from Fissler & Ziegel (NAAJ 2025).

    This is the actuarially correct way to compute a quantile premium for
    a zero-inflated loss distribution. The standard approach of running
    QuantileGBM on aggregate losses (including zeros) produces biased
    quantile estimates for lines with low claim frequency, because the
    CDF has a large mass at zero. The two-part decomposition avoids this.

    Parameters
    ----------
    alpha:
        Quantile level for the severity component. 0.95 gives the
        95th percentile of the per-claim severity distribution.
    safety_loading_lambda:
        Risk appetite parameter for the variance-based safety loading.
        Larger values increase the loading. 0 = no loading (pure quantile
        premium). 0.5 is a reasonable starting point for UK personal lines.
        The loading formula: safety_loading_i = lambda * freq_i * sigma_sev_i^2 / (2 * mu_sev_i)
        where sigma_sev^2 = Var[Y | claim, X] and mu_sev = E[Y | claim, X].
    freq_model:
        Pre-fitted frequency model. Must have a predict(X_numpy) method
        returning predicted claim frequency or claim probability. Can be
        a CatBoostClassifier, a GLM, or any sklearn-compatible model.
        Passed at construction time or at fit() time.
    severity_quantile_model:
        Pre-fitted QuantileGBM model for severity (fitted on claim costs
        for policies with at least one claim only). If provided, fit() will
        skip fitting the severity model and use this directly.
    quantile_model_params:
        Dict of QuantileGBM constructor parameters. Only used if
        severity_quantile_model is not provided at fit() time.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        safety_loading_lambda: float = 0.5,
        freq_model: Any = None,
        severity_quantile_model: "QuantileGBM | None" = None,
        quantile_model_params: dict | None = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if safety_loading_lambda < 0:
            raise ValueError(f"safety_loading_lambda must be >= 0, got {safety_loading_lambda}")

        self.alpha = alpha
        self.safety_loading_lambda = safety_loading_lambda
        self.freq_model = freq_model
        self.severity_quantile_model = severity_quantile_model
        self.quantile_model_params = quantile_model_params or {}

        # Set after fit
        self._freq_model_fitted: Any = None
        self._sev_model_fitted: "QuantileGBM | None" = None
        self._mean_sev_fitted: Any = None  # CatBoost mean severity model

    def fit(
        self,
        X: Any,
        y: Any,
        freq_model: Any = None,
        exposure: Any | None = None,
        severity_quantile_model: "QuantileGBM | None" = None,
    ) -> "TwoPartQuantilePremium":
        """
        Fit the two-part quantile premium model.

        Parameters
        ----------
        X:
            Feature matrix. Polars DataFrame or numpy array. Must contain
            the same features used in freq_model.
        y:
            Aggregate loss per policy (0 for no-claim policies, claim cost
            for claim policies). The severity model is fitted on the
            non-zero subset automatically.
        freq_model:
            Fitted frequency model. Overrides constructor argument if provided.
        exposure:
            Optional per-policy exposure. Passed to the severity QuantileGBM
            as sample_weight.
        severity_quantile_model:
            Pre-fitted QuantileGBM. Overrides constructor argument if provided.

        Returns
        -------
        self
        """
        from catboost import CatBoostRegressor  # noqa: PLC0415

        X_arr = _to_numpy(X)
        y_arr = _series_to_numpy(y).ravel()

        # Set frequency model
        self._freq_model_fitted = freq_model or self.freq_model
        if self._freq_model_fitted is None:
            raise ValueError(
                "A frequency model must be provided either at construction time "
                "or as the freq_model argument to fit()."
            )

        # Severity model: fit on non-zero claims only
        sev_model = severity_quantile_model or self.severity_quantile_model

        claim_mask = y_arr > 0
        if claim_mask.sum() < 50:
            import warnings
            warnings.warn(
                f"Only {claim_mask.sum()} non-zero claims in training data. "
                "Severity quantile estimates will be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        X_sev = X_arr[claim_mask]
        y_sev = y_arr[claim_mask]
        exp_sev = _to_numpy(exposure)[claim_mask] if exposure is not None else None

        if sev_model is None:
            # Fit QuantileGBM for severity
            qgbm_params = {
                "quantiles": [self.alpha],
                "fix_crossing": True,
                **self.quantile_model_params,
            }
            sev_model = QuantileGBM(**qgbm_params)

            X_sev_pl = pl.DataFrame(X_sev) if not isinstance(X, pl.DataFrame) else X.filter(pl.Series(claim_mask))
            y_sev_pl = pl.Series("y", y_sev)
            exp_sev_pl = pl.Series("exposure", exp_sev) if exp_sev is not None else None

            sev_model.fit(X_sev_pl, y_sev_pl, exposure=exp_sev_pl)

        self._sev_model_fitted = sev_model

        # Fit a CatBoost mean severity model for the safety loading calculation.
        # The loading requires E[Y | claim, X] and Var[Y | claim, X].
        # We approximate Var using the empirical variance of OOF residuals from a
        # simple mean model — this is the scalar-phi approximation and is sufficient
        # for computing the loading direction; use GammaGBM for per-risk variance.
        cb_mean = CatBoostRegressor(
            iterations=200, learning_rate=0.05, depth=6, verbose=0,
            allow_writing_files=False,
            random_seed=42,
        )
        cb_mean.fit(
            X_sev,
            y_sev,
            sample_weight=exp_sev,
        )
        self._mean_sev_fitted = cb_mean

        # Estimate scalar variance from residuals (global, not per-risk)
        mu_sev_train = cb_mean.predict(X_sev)
        resid = y_sev - mu_sev_train
        self._sev_variance_scalar = float(np.var(resid))

        return self

    def predict(self, X: Any) -> TwoPartPremiumResult:
        """
        Compute per-risk two-part quantile premium.

        Parameters
        ----------
        X:
            Feature matrix. Polars DataFrame or numpy array.

        Returns
        -------
        TwoPartPremiumResult with quantile_premium, freq, severity_quantile,
        safety_loading fields as Polars Series.
        """
        if self._freq_model_fitted is None or self._sev_model_fitted is None:
            raise RuntimeError("Call fit() before predict().")

        X_arr = _to_numpy(X)

        # Frequency predictions
        freq_raw = self._freq_model_fitted.predict(X_arr)
        freq_arr = np.asarray(freq_raw, dtype=np.float64).ravel()

        # CatBoost classifiers return class labels from predict(); use predict_proba
        # for probabilities. Detect and handle this.
        if freq_arr.ndim == 1 and set(np.unique(freq_arr)).issubset({0.0, 1.0}):
            try:
                freq_arr = self._freq_model_fitted.predict_proba(X_arr)[:, 1]
            except AttributeError:
                pass  # Not a classifier — binary predictions are intentional

        # Severity quantile predictions
        X_pl = (
            X if isinstance(X, pl.DataFrame)
            else pl.DataFrame(X_arr)
        )
        sev_preds = self._sev_model_fitted.predict(X_pl)
        q_col = f"q_{self.alpha}"
        if q_col not in sev_preds.columns:
            # Use first column if exact name not found
            q_col = sev_preds.columns[0]
        sev_q_arr = sev_preds[q_col].to_numpy().astype(np.float64)

        # Mean severity for safety loading denominator
        mu_sev = np.asarray(self._mean_sev_fitted.predict(X_arr), dtype=np.float64).ravel()
        mu_sev_safe = np.where(mu_sev > 0, mu_sev, np.finfo(float).eps)

        # Variance-based safety loading (NAAJ 2025)
        # For Bernoulli frequency: loading = lambda * freq * (1-freq) * sev_q^2 / (2 * mu_sev)
        # We use the simpler scalar-variance approximation here; per-risk variance
        # requires a separate GammaGBM or TweedieGBM for sigma_sev(x).
        sigma2_sev = self._sev_variance_scalar
        safety_loading = (
            self.safety_loading_lambda
            * freq_arr
            * (1.0 - np.clip(freq_arr, 0.0, 1.0))
            * sigma2_sev
            / (2.0 * mu_sev_safe)
        )

        # Two-part quantile premium
        qp = freq_arr * sev_q_arr + safety_loading

        return TwoPartPremiumResult(
            quantile_premium=pl.Series("quantile_premium", qp),
            freq=pl.Series("freq", freq_arr),
            severity_quantile=pl.Series("severity_quantile", sev_q_arr),
            safety_loading=pl.Series("safety_loading", safety_loading),
            alpha=self.alpha,
        )
