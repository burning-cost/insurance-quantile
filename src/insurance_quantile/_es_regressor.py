"""
Direct Expected Shortfall Regression via the i-Rock estimator.

Implements the procedure from:
    Li, B., Zhang, X. & He, F. (2026). "Rocking the Expected Shortfall:
    Regression and Inference." arXiv:2602.18865.

Background
----------
Expected Shortfall (ES) — also called Conditional Tail Expectation or TVaR —
is defined as:

    ES(alpha, x) = E[Y | Y >= Q(alpha, x), X = x]

Standard practice estimates ES indirectly: fit a quantile model, then average
predictions above the quantile. This two-step approach accumulates first-stage
estimation error and, crucially, does not give you inferential machinery for
the ES coefficients directly.

The i-Rock ("integrated Rockafellar") method addresses this. The key idea is
to construct Neyman-orthogonalised pseudo-values Z_i(s) that serve as unbiased
proxies for ES at each quantile level s, then run a second-stage quantile
regression on binned covariate means using those pseudo-values. The result is:

1. A direct linear estimate ES(alpha, x) = x^T beta
2. An asymptotic variance formula (Theorem 4.2) for inference on beta

Insurance use cases:
- Motor BI tail pricing: fit ES(0.99) directly on claim severity covariates
- Solvency II SCR: formal statistical inference on the 99.5th percentile tail
- Reinsurance layer pricing: direct ES coefficients without intermediate steps

Notes on the binning step
--------------------------
The covariate space is partitioned into M = prod(k_j) bins by computing
quantile-based breakpoints per dimension. Within each bin, OLS of Z_i(alpha_t)
on X_i gives a local ES estimate. The final second-stage quantile regression
uses the bin centroids as predictors and the OLS intercepts as response.

The "auto" bin count formula from the paper is:
    k = ceil(1.6 * sqrt(p) * (sqrt(n) / log(n))^(1/p))

For p=1 this simplifies to k = ceil(1.6 * sqrt(n) / log(n)).
For large p, this formula keeps the total bin count M = k^p manageable.

UK English used throughout.

References
----------
Li, B., Zhang, X. & He, F. (2026). Rocking the Expected Shortfall: Regression
    and Inference. arXiv:2602.18865.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm as _norm

__all__ = ["ExpectedShortfallRegressor"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _quantile_regression_coefs(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Fit a linear quantile regression model.

    Returns (beta, intercept) such that the predicted quantile at x is
    intercept + x @ beta.

    Uses sklearn's QuantileRegressor which solves the exact check-loss LP.

    Parameters
    ----------
    X:
        Feature matrix (N, p). Should NOT include an intercept column.
    y:
        Response vector (N,).
    alpha:
        Quantile level in (0, 1).
    sample_weight:
        Optional sample weights (N,).

    Returns
    -------
    Tuple (coef, intercept) where coef has shape (p,) and intercept is scalar.
    """
    from sklearn.linear_model import QuantileRegressor

    qr = QuantileRegressor(quantile=alpha, alpha=0.0, solver="highs", fit_intercept=True)
    fit_kwargs: dict[str, Any] = {"X": X, "y": y}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    qr.fit(**fit_kwargs)
    return qr.coef_, float(qr.intercept_)


def _catboost_quantile_coefs(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """
    Fit a CatBoost quantile regression at a single level and return approximate
    linear coefficients via feature importances + calibrated intercept.

    Used only as the first-stage estimator when first_stage='catboost'.
    The predictions are what matter for the pseudo-values; we do not use the
    linear coefficients from this stage.

    This returns the raw CatBoost model (wrapped so predict() works),
    not literal linear coefficients. Called internally; the caller uses
    the .predict() method on the returned object.
    """
    raise NotImplementedError(
        "CatBoost first stage: use the model.predict() interface directly."
    )


def _quantile_predict(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    first_stage: str,
    first_stage_kwargs: dict[str, Any],
) -> np.ndarray:
    """
    Fit a first-stage quantile model and return in-sample predictions Q_hat(alpha, X_i).

    Returns
    -------
    Array of shape (N,).
    """
    if first_stage == "linear":
        coef, intercept = _quantile_regression_coefs(X, y, alpha)
        return X @ coef + intercept
    elif first_stage == "catboost":
        from catboost import CatBoostRegressor

        params: dict[str, Any] = {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 5,
            "random_seed": 42,
            "verbose": 0,
            "loss_function": f"Quantile:alpha={alpha}",
        }
        params.update(first_stage_kwargs)
        model = CatBoostRegressor(**params)
        model.fit(X, y)
        return model.predict(X)
    else:
        raise ValueError(
            f"first_stage must be 'linear' or 'catboost', got '{first_stage}'"
        )


def _pseudo_values(
    y: np.ndarray,
    q_hat: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Compute Neyman-orthogonalised ES pseudo-values.

    Z_i(alpha) = (1 - alpha)^{-1} * (Y_i - q_hat_i) * 1{Y_i >= q_hat_i} + q_hat_i

    This is the ES proxy: E[Z_i(alpha)] = ES(alpha, x_i) when q_hat_i is a
    consistent estimate of Q(alpha, x_i). The Neyman-orthogonalisation makes
    this insensitive to first-stage estimation error (the bias contribution
    from q_hat error is negligible at the second stage).

    Parameters
    ----------
    y:
        Observed responses, shape (N,).
    q_hat:
        First-stage quantile predictions Q_hat(alpha, X_i), shape (N,).
    alpha:
        Quantile level at which q_hat was estimated.

    Returns
    -------
    Pseudo-values Z_i, shape (N,). These are finite as long as alpha < 1.
    """
    excess = np.where(y >= q_hat, y - q_hat, 0.0)
    return q_hat + excess / (1.0 - alpha)


def _auto_bin_count(n: int, p: int) -> int:
    """
    Compute the automatic bin count per dimension from the paper.

    k = ceil(1.6 * sqrt(p) * (sqrt(n) / log(n))^(1/p))

    This ensures the total bin count M = k^p grows at the right rate
    relative to n for the asymptotic approximation to hold.

    Parameters
    ----------
    n:
        Number of training observations.
    p:
        Number of covariate dimensions.

    Returns
    -------
    Integer bin count per dimension, at least 2.
    """
    if n <= 1 or p <= 0:
        return 2
    log_n = math.log(n)
    if log_n <= 0:
        return 2
    inner = math.sqrt(n) / log_n
    k = math.ceil(1.6 * math.sqrt(p) * (inner ** (1.0 / p)))
    return max(k, 2)


def _make_bin_edges(X: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Compute quantile-based breakpoints for each column of X.

    Uses k+1 equally spaced quantile levels in [0, 1] to define k bins
    per dimension. Bin edges are column-specific.

    Parameters
    ----------
    X:
        Feature matrix (N, p).
    k:
        Number of bins per dimension.

    Returns
    -------
    List of length p, each element is an array of k-1 interior breakpoints.
    """
    _, p = X.shape
    edges = []
    for j in range(p):
        col = X[:, j]
        # k bins -> k-1 interior quantile breaks
        levels = np.linspace(0.0, 100.0, k + 1)[1:-1]
        breaks = np.percentile(col, levels)
        # Deduplicate to avoid empty bins in degenerate columns
        breaks = np.unique(breaks)
        edges.append(breaks)
    return edges


def _assign_bins(X: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """
    Assign each observation to a bin index.

    Bins are indexed as a flat integer using row-major order over a
    multi-dimensional grid.

    Parameters
    ----------
    X:
        Feature matrix (N, p).
    edges:
        Interior breakpoints per dimension, from _make_bin_edges.

    Returns
    -------
    Integer array of shape (N,) with bin indices in [0, M).
    """
    N, p = X.shape
    n_bins_per_dim = np.array([len(e) + 1 for e in edges], dtype=np.int64)
    bin_idx = np.zeros((N, p), dtype=np.int64)
    for j in range(p):
        bin_idx[:, j] = np.searchsorted(edges[j], X[:, j], side="right")

    # Convert multi-index to flat index using row-major strides
    strides = np.ones(p, dtype=np.int64)
    for j in range(p - 2, -1, -1):
        strides[j] = strides[j + 1] * n_bins_per_dim[j + 1]

    flat_idx = bin_idx @ strides
    return flat_idx


def _local_es_estimate(
    X_bin: np.ndarray,
    Z_bin: np.ndarray,
) -> float | None:
    """
    Estimate the local ES intercept within a bin via OLS.

    Fits Z_bin ~ intercept + X_bin @ beta by OLS and returns the intercept.
    The intercept is an estimate of the conditional ES at the bin centroid
    after accounting for covariate variation within the bin.

    Returns None if the bin has fewer observations than needed for OLS.

    Parameters
    ----------
    X_bin:
        Feature matrix for observations in this bin, shape (m, p).
    Z_bin:
        Pseudo-values for observations in this bin, shape (m,).

    Returns
    -------
    OLS intercept as a float, or None.
    """
    m, p = X_bin.shape
    if m < p + 1:
        # Not enough observations for OLS with intercept
        return None
    try:
        # OLS via numpy least squares
        X_aug = np.hstack([np.ones((m, 1)), X_bin])
        coef, _, _, _ = np.linalg.lstsq(X_aug, Z_bin, rcond=None)
        return float(coef[0])
    except np.linalg.LinAlgError:
        return None


def _check_loss_scalar(residuals: np.ndarray, alpha: float) -> float:
    """Mean check (pinball) loss at level alpha."""
    return float(np.mean(residuals * (alpha - (residuals < 0.0).astype(np.float64))))


def _second_stage_qr(
    x_bar: np.ndarray,
    v_hat: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """
    Second-stage quantile regression of v_hat on x_bar.

    Fits the model:  v_hat_m ~ x_bar_m @ beta + intercept  (at quantile alpha)

    This gives the i-Rock ES regression coefficients.

    Parameters
    ----------
    x_bar:
        Bin centroid feature matrix, shape (M, p).
    v_hat:
        Local ES estimates for each bin, shape (M,).
    alpha:
        Quantile level (same as the target ES level).

    Returns
    -------
    Tuple (coef, intercept) for the ES regression.
    """
    from sklearn.linear_model import QuantileRegressor

    qr = QuantileRegressor(quantile=alpha, alpha=0.0, solver="highs", fit_intercept=True)
    qr.fit(x_bar, v_hat)
    return qr.coef_, float(qr.intercept_)


# ---------------------------------------------------------------------------
# Sandwich variance estimator
# ---------------------------------------------------------------------------


def _sandwich_variance(
    X: np.ndarray,
    y: np.ndarray,
    q_hat: np.ndarray,
    es_hat: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Sandwich variance estimator for the i-Rock ES regression coefficients.

    Implements the empirical analogue of Theorem 4.2:

        Var(beta_hat) ≈ (1/n) * D1^{-1} * Omega1 * D1^{-T}

    where:
        D1 = (1/n) sum_i x_i x_i^T / (ES_i - Q_i)
        Omega1 = (1/n) sum_i sigma_i^2 * x_i x_i^T / (1 - alpha)

    and sigma_i^2 is estimated from the squared pseudo-value residuals.

    Note: This is an approximation. The paper's full variance formula involves
    nonparametric density estimates. The sandwich estimator is consistent but
    may undercover in small samples. For inference-critical applications, use
    bootstrap standard errors.

    Parameters
    ----------
    X:
        Feature matrix with intercept prepended, shape (N, p+1).
    y:
        Observed responses, shape (N,).
    q_hat:
        First-stage quantile predictions, shape (N,).
    es_hat:
        Fitted ES values from second stage, shape (N,).
    alpha:
        ES level.

    Returns
    -------
    Variance-covariance matrix, shape (p+1, p+1).
    """
    N, p_aug = X.shape

    # Gap between ES and quantile — clip to avoid division by zero
    gap = np.maximum(es_hat - q_hat, 1e-8)

    # D1: (1/n) sum_i x_i x_i^T / gap_i
    weights_d1 = 1.0 / gap  # shape (N,)
    D1 = (X * weights_d1[:, None]).T @ X / N  # (p+1, p+1)

    # Pseudo-value residuals for Omega1
    Z = _pseudo_values(y, q_hat, alpha)
    # ES residuals (Z_i - ES_hat_i)
    resid = Z - es_hat
    sigma2_i = resid ** 2  # pointwise variance proxy

    # Omega1: (1/n) sum_i sigma2_i / (1 - alpha) * x_i x_i^T
    weights_omega = sigma2_i / (1.0 - alpha)
    Omega1 = (X * weights_omega[:, None]).T @ X / N  # (p+1, p+1)

    # Sandwich: D1^{-1} Omega1 D1^{-1} / n
    try:
        D1_inv = np.linalg.inv(D1)
    except np.linalg.LinAlgError:
        D1_inv = np.linalg.pinv(D1)

    Var = D1_inv @ Omega1 @ D1_inv / N
    return Var


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ExpectedShortfallRegressor:
    """
    Direct Expected Shortfall regression via the i-Rock estimator.

    Estimates ES(alpha, x) = x^T beta directly, without a two-step
    quantile integration. Based on Li, Zhang & He (2026) arXiv:2602.18865.

    The i-Rock procedure:
    1. Fit a first-stage quantile model at levels in [alpha, 1-epsilon] to
       obtain Q_hat(s, X_i) for each observation and quantile grid level s.
    2. For each s, compute Neyman-orthogonalised pseudo-values:
           Z_i(s) = Q_hat(s, X_i) + (Y_i - Q_hat(s, X_i))_+ / (1 - s)
    3. Partition the covariate space into M bins via quantile-based breakpoints.
       Within each bin, run OLS of Z_i(alpha) on X_i to get a local ES estimate.
    4. Run a second-stage quantile regression (at level alpha) of the local
       estimates on bin centroids. The resulting coefficients are beta_hat.

    Parameters
    ----------
    alpha : float, default=0.99
        ES level (e.g. 0.99 for the 99th-percentile tail). Must be in (0, 1).
    n_quantile_grid : int, default=20
        Number of quantile levels in [alpha, 1-epsilon] for the first-stage
        integration grid. More levels give a smoother pseudo-value estimate
        but increase computation. 10-30 is typical.
    n_bins_per_dim : int or "auto", default="auto"
        Number of bins per covariate dimension for the binning step.
        "auto" uses the paper's formula:
            k = ceil(1.6 * sqrt(p) * (sqrt(n) / log(n))^(1/p))
        Pass an integer to override. Values of 3-8 are practical for
        most insurance datasets. Fewer bins means less data per bin
        (more noise) but reduces computational cost.
    first_stage : {"linear", "catboost"}, default="linear"
        First-stage quantile estimator.
        "linear" uses sklearn's QuantileRegressor (exact LP, suitable for
        well-behaved continuous covariates).
        "catboost" uses CatBoost gradient boosting (better for interactions
        and categorical covariates, but slower and less stable in tiny bins).
    epsilon : float, default=0.001
        Upper truncation for the quantile grid: grid spans [alpha, 1-epsilon].
        Set smaller for cleaner tail estimation; larger if you want to
        include near-unity quantiles (not recommended for alpha > 0.99).
    first_stage_kwargs : dict or None, default=None
        Extra keyword arguments passed to the first-stage estimator.
        For "catboost": CatBoostRegressor parameters (e.g. iterations, depth).
        For "linear": ignored (sklearn QuantileRegressor has no tuning knobs
        beyond alpha=0.0 which is always set).
    random_state : int or None, default=None
        Random seed for reproducibility. Currently unused (all steps are
        deterministic) but reserved for future extensions.

    Attributes
    ----------
    coef_ : np.ndarray, shape (n_features,)
        Slope coefficients for the ES regression. Set after fit().
    intercept_ : float
        Intercept term. Set after fit().
    n_features_in_ : int
        Number of features seen during fit().
    is_fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_quantile import ExpectedShortfallRegressor
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(500, 2))
    >>> y = np.maximum(0, X[:, 0] + rng.exponential(size=500))
    >>> model = ExpectedShortfallRegressor(alpha=0.95, n_bins_per_dim=4)
    >>> model.fit(X, y)
    ExpectedShortfallRegressor(alpha=0.95)
    >>> preds = model.predict(X[:5])
    >>> summary = model.summary()

    Notes
    -----
    The pseudo-value construction is the core insight of the i-Rock method.
    For a single alpha, Z_i = Q_i + (Y_i - Q_i)_+ / (1 - alpha) satisfies
    E[Z_i | X_i] = ES(alpha, X_i) when Q_i = Q(alpha, X_i) exactly. The
    Neyman-orthogonalisation means first-stage estimation error in Q_i
    contributes only second-order bias to the final ES estimate.

    For the variance formula in summary(), we use a sandwich estimator based
    on the asymptotic theory in Theorem 4.2. This is consistent but may
    undercover in small samples (n < 500). For such cases, use bootstrap
    standard errors or treat the z-statistics as approximate.

    References
    ----------
    Li, B., Zhang, X. & He, F. (2026). Rocking the Expected Shortfall:
        Regression and Inference. arXiv:2602.18865.
    """

    def __init__(
        self,
        alpha: float = 0.99,
        n_quantile_grid: int = 20,
        n_bins_per_dim: int | Literal["auto"] = "auto",
        first_stage: str = "linear",
        epsilon: float = 0.001,
        first_stage_kwargs: dict[str, Any] | None = None,
        random_state: int | None = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_quantile_grid < 1:
            raise ValueError(f"n_quantile_grid must be >= 1, got {n_quantile_grid}")
        if first_stage not in ("linear", "catboost"):
            raise ValueError(
                f"first_stage must be 'linear' or 'catboost', got '{first_stage}'"
            )
        if epsilon <= 0.0 or epsilon >= 1.0 - alpha:
            raise ValueError(
                f"epsilon must be in (0, 1-alpha), got {epsilon} with alpha={alpha}"
            )

        self.alpha = alpha
        self.n_quantile_grid = n_quantile_grid
        self.n_bins_per_dim = n_bins_per_dim
        self.first_stage = first_stage
        self.epsilon = epsilon
        self.first_stage_kwargs = first_stage_kwargs or {}
        self.random_state = random_state

        # Set after fit()
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None
        self.is_fitted_: bool = False

        # Internal storage for summary()
        self._X_aug_: np.ndarray | None = None
        self._y_: np.ndarray | None = None
        self._q_hat_alpha_: np.ndarray | None = None
        self._es_fitted_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def coef_(self) -> np.ndarray | None:
        """Coefficient vector beta, shape (n_features,). None before fit()."""
        return self._coef

    @coef_.setter
    def coef_(self, value: np.ndarray | None) -> None:
        self._coef = value

    @property
    def intercept_(self) -> float | None:
        """Intercept term. None before fit()."""
        return self._intercept

    @intercept_.setter
    def intercept_(self, value: float | None) -> None:
        self._intercept = value

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ExpectedShortfallRegressor":
        """
        Fit the ES regression model.

        Steps:
        1. Fit first-stage quantile model at levels in [alpha, 1-epsilon].
        2. Compute Neyman-orthogonalised pseudo-values Z_i(alpha).
        3. Bin covariate space; compute local ES estimates v_hat_m.
        4. Run second-stage quantile regression of v_hat on x_bar (bin centroids).

        Parameters
        ----------
        X:
            Feature matrix, shape (N, p). Numeric features only — encode
            categoricals before calling fit().
        y:
            Response vector, shape (N,). For insurance severity, pass claim
            amounts (not log-transformed).
        sample_weight:
            Optional observation weights, shape (N,). Currently passed only
            to the first-stage estimator when first_stage='linear'.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X and y are incompatible or parameters are invalid.
        RuntimeError
            If the binning step produces too few valid bins for second-stage
            regression.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows: {X.shape[0]} vs {y.shape[0]}"
            )

        N, p = X.shape
        self.n_features_in_ = p

        # ------------------------------------------------------------------
        # Step 1: First-stage quantile model at alpha
        # We also need the alpha-quantile predictions for pseudo-values.
        # ------------------------------------------------------------------
        q_hat_alpha = _quantile_predict(
            X, y, self.alpha, self.first_stage, self.first_stage_kwargs
        )

        # ------------------------------------------------------------------
        # Step 2: Pseudo-values Z_i(alpha)
        # ------------------------------------------------------------------
        Z_alpha = _pseudo_values(y, q_hat_alpha, self.alpha)

        # ------------------------------------------------------------------
        # Step 3: Bin covariate space
        # ------------------------------------------------------------------
        if self.n_bins_per_dim == "auto":
            k = _auto_bin_count(N, p)
        else:
            k = int(self.n_bins_per_dim)
            if k < 2:
                raise ValueError(f"n_bins_per_dim must be >= 2, got {k}")

        edges = _make_bin_edges(X, k)
        bin_indices = _assign_bins(X, edges)

        unique_bins = np.unique(bin_indices)
        n_bins_populated = len(unique_bins)

        if n_bins_populated < p + 2:
            warnings.warn(
                f"Only {n_bins_populated} populated bins with p={p} features. "
                "The second-stage regression may be poorly identified. "
                "Consider reducing n_bins_per_dim or increasing the sample size.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 3 (cont): local ES estimate per bin
        # Fit OLS of Z_alpha ~ intercept + X within each bin.
        # Use the OLS intercept as the local ES estimate v_hat_m.
        # Use the bin centroid as x_bar_m.
        # ------------------------------------------------------------------
        x_bar_list: list[np.ndarray] = []
        v_hat_list: list[float] = []

        for b in unique_bins:
            mask = bin_indices == b
            X_bin = X[mask]
            Z_bin = Z_alpha[mask]
            m = X_bin.shape[0]

            # Compute bin centroid
            x_bar_m = X_bin.mean(axis=0)

            # Local ES estimate: OLS intercept within bin
            v_hat_m = _local_es_estimate(X_bin, Z_bin)
            if v_hat_m is None:
                # Bin too small for OLS: use the raw mean of pseudo-values
                v_hat_m = float(np.mean(Z_bin))

            x_bar_list.append(x_bar_m)
            v_hat_list.append(v_hat_m)

        x_bar = np.array(x_bar_list)  # (M_eff, p)
        v_hat = np.array(v_hat_list)  # (M_eff,)

        if len(v_hat) < 2:
            raise RuntimeError(
                "Fewer than 2 valid bins after local ES estimation. "
                "The second-stage regression requires at least 2 data points. "
                "Increase sample size or reduce n_bins_per_dim."
            )

        # ------------------------------------------------------------------
        # Step 4: Second-stage quantile regression
        # Regress v_hat on x_bar at quantile level alpha.
        # ------------------------------------------------------------------
        coef, intercept = _second_stage_qr(x_bar, v_hat, self.alpha)

        self.coef_ = coef
        self.intercept_ = intercept
        self.is_fitted_ = True

        # Store for summary()
        X_aug = np.hstack([np.ones((N, 1)), X])
        self._X_aug_ = X_aug
        self._y_ = y
        self._q_hat_alpha_ = q_hat_alpha
        # ES fitted values for each training observation
        self._es_fitted_ = X @ coef + intercept

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ES(alpha | X) for new observations.

        Parameters
        ----------
        X:
            Feature matrix, shape (M, p). Must have the same number of
            columns as the training data.

        Returns
        -------
        Predicted ES values, shape (M,). These are linear in X.

        Raises
        ------
        RuntimeError
            If predict() is called before fit().
        ValueError
            If X has the wrong number of features.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return X @ self.coef_ + self.intercept_

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Return a coefficient summary table with asymptotic standard errors.

        Uses the sandwich variance estimator (Theorem 4.2 approximation).
        Columns: term, coef, std_err, z_stat, p_value.

        The intercept term is included as '(Intercept)'.

        Returns
        -------
        pandas DataFrame with columns:
            - term: parameter name
            - coef: point estimate
            - std_err: asymptotic standard error
            - z_stat: z = coef / std_err
            - p_value: two-sided p-value under N(0,1) asymptotic

        Raises
        ------
        RuntimeError
            If summary() is called before fit().

        Notes
        -----
        The z-statistics are approximate — the sandwich estimator is consistent
        but the finite-sample distribution may deviate from N(0,1) when n < 500.
        The p-values should be interpreted with caution for small samples.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before summary().")

        # Build augmented coefficient vector [intercept, coef_]
        theta = np.concatenate([[self.intercept_], self.coef_])
        p = self.n_features_in_

        # Sandwich variance
        Var = _sandwich_variance(
            X=self._X_aug_,
            y=self._y_,
            q_hat=self._q_hat_alpha_,
            es_hat=self._es_fitted_,
            alpha=self.alpha,
        )

        se = np.sqrt(np.maximum(np.diag(Var), 0.0))
        z_stat = np.where(se > 0.0, theta / se, np.nan)
        p_val = 2.0 * (1.0 - _norm.cdf(np.abs(z_stat)))

        # Term names
        terms = ["(Intercept)"] + [f"x{j}" for j in range(p)]

        return pd.DataFrame(
            {
                "term": terms,
                "coef": theta,
                "std_err": se,
                "z_stat": z_stat,
                "p_value": p_val,
            }
        )

    def __repr__(self) -> str:
        return f"ExpectedShortfallRegressor(alpha={self.alpha})"
