"""
Wasserstein Distributionally Robust Quantile Regression (WDRQR).

Implements the closed-form estimator from:
    Zhang, Mao & Wang (2026). "Wasserstein Distributionally Robust Quantile
    Regression." arXiv:2603.14991.

The key result (Theorem 1): for p >= 2 and a p-Wasserstein ambiguity ball of
radius eps around the empirical measure, the worst-case quantile regression
problem reduces to standard QR with (a) an L2 slope penalty and (b) an analytic
intercept correction. The slope penalty coefficient and intercept correction are
both closed-form in tau, eps and ||beta||.

For p = 1 (W_1 distance), the estimator is identical to standard QR regardless
of eps. W_1 robustness is already implicit in the check loss — there is no
benefit to using WDRQR with p = 1.

Insurance use cases:
- Reserve quantile for thin-data segments (N < 500, tau >= 0.95)
- Large loss loading under distribution shift uncertainty
- Per-risk Solvency II SCR quantile with formal out-of-sample guarantee

UK English used throughout.

References
----------
Zhang, C., Mao, T., & Wang, R. (2026). Wasserstein Distributionally Robust
    Quantile Regression. arXiv:2603.14991.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize

__all__ = [
    "WassersteinRobustQR",
    "wdrqr_large_loss_loading",
    "wdrqr_reserve_quantile",
]


# ---------------------------------------------------------------------------
# Core mathematics (numpy / scipy only — no CVXPY)
# ---------------------------------------------------------------------------


def _check_loss(residuals: np.ndarray, tau: float) -> float:
    """
    Pinball (check) loss for a vector of residuals.

    rho_tau(u) = u * (tau - I(u < 0))

    Parameters
    ----------
    residuals:
        y - y_hat, shape (N,).
    tau:
        Quantile level in (0, 1).

    Returns
    -------
    Mean check loss.
    """
    return float(np.mean(residuals * (tau - (residuals < 0).astype(np.float64))))


def _c_tau_p(tau: float, p: int) -> float:
    """
    Closed-form constant from Theorem 1 of Zhang et al. (2026).

    c_{tau,p} = (tau^q + (1-tau)^q)^(1/q)

    where q = p / (p - 1) is the conjugate exponent of p.

    For p = 2, q = 2 and c_{tau,2} = sqrt(tau^2 + (1-tau)^2).

    Parameters
    ----------
    tau:
        Quantile level.
    p:
        Wasserstein ball type. Must be >= 1.

    Returns
    -------
    Positive constant c_{tau,p}.
    """
    if p == 1:
        # Limit: q -> inf, so (tau^q + (1-tau)^q)^(1/q) -> max(tau, 1-tau)
        return max(tau, 1.0 - tau)
    q = p / (p - 1)
    return float((tau**q + (1.0 - tau) ** q) ** (1.0 / q))


def _intercept_correction(
    beta_norm: float,
    tau: float,
    p: int,
    eps: float,
) -> float:
    """
    Theorem 1 intercept correction term.

    s* = s_bar* + (eps / q) * (tau^q - (1-tau)^q) * c_{tau,p}^(1-q) * ||beta*||

    For tau > 0.5 and eps > 0, this correction is positive — it shifts the
    predicted quantile upward, correcting for the systematic downward bias of
    empirical high quantiles in small samples.

    Parameters
    ----------
    beta_norm:
        L2 norm of the fitted slope vector ||beta*||.
    tau:
        Quantile level.
    p:
        Wasserstein ball type.
    eps:
        Wasserstein ball radius.

    Returns
    -------
    Scalar intercept correction (positive for tau > 0.5, eps > 0).
    """
    if p == 1 or eps == 0.0:
        return 0.0
    q = p / (p - 1)
    c = _c_tau_p(tau, p)
    # When beta is zero, correction is zero regardless
    if beta_norm == 0.0:
        return 0.0
    correction = (eps / q) * (tau**q - (1.0 - tau) ** q) * (c ** (1.0 - q)) * beta_norm
    return float(correction)


def _fit_slope_and_intercept(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    p: int,
    eps: float,
) -> tuple[np.ndarray, float]:
    """
    Fit the robust slope and intercept via L-BFGS-B.

    The slope is the minimiser of:
        E_hat[rho_tau(y - X beta - s_bar)] + c_{tau,p} * eps * ||beta||_2

    where s_bar is estimated analytically as the (weighted) sample quantile
    of residuals for the current beta at each optimisation step.

    After convergence, the intercept is corrected using Theorem 1.

    Parameters
    ----------
    X:
        Feature matrix, shape (N, d). Should NOT include an intercept column.
    y:
        Response vector, shape (N,).
    tau:
        Quantile level.
    p:
        Wasserstein ball type.
    eps:
        Wasserstein ball radius.

    Returns
    -------
    Tuple of (beta, s_star) where beta has shape (d,) and s_star is a scalar.
    """
    N, d = X.shape
    c_penalty = _c_tau_p(tau, p) * eps

    def objective(beta: np.ndarray) -> float:
        # Estimate intercept analytically: it's the tau-quantile of (y - X beta)
        residuals_no_intercept = y - X @ beta
        s_bar = float(np.quantile(residuals_no_intercept, tau))
        residuals = residuals_no_intercept - s_bar
        loss = _check_loss(residuals, tau)
        reg = c_penalty * float(np.linalg.norm(beta))
        return loss + reg

    def gradient(beta: np.ndarray) -> np.ndarray:
        # Sub-gradient: dL/d_beta + c * eps * beta / ||beta|| (if beta != 0)
        residuals_no_intercept = y - X @ beta
        s_bar = float(np.quantile(residuals_no_intercept, tau))
        residuals = residuals_no_intercept - s_bar
        # Sub-gradient of check loss w.r.t. beta:
        # d rho_tau(r_i) / d beta_j = -(tau - I(r_i < 0)) * X_{ij}
        weights = tau - (residuals < 0).astype(np.float64)
        grad_loss = -X.T @ weights / N
        # Sub-gradient of L2 norm
        norm_beta = float(np.linalg.norm(beta))
        if norm_beta > 1e-12:
            grad_reg = c_penalty * beta / norm_beta
        else:
            grad_reg = np.zeros(d)
        return grad_loss + grad_reg

    beta0 = np.zeros(d)
    result = minimize(
        objective,
        beta0,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )

    beta_star = result.x

    # Analytic intercept: tau-quantile of (y - X beta*)
    s_bar_star = float(np.quantile(y - X @ beta_star, tau))

    # Theorem 1 intercept correction
    beta_norm = float(np.linalg.norm(beta_star))
    correction = _intercept_correction(beta_norm, tau, p, eps)
    s_star = s_bar_star + correction

    return beta_star, s_star


# ---------------------------------------------------------------------------
# Public API: WassersteinRobustQR class
# ---------------------------------------------------------------------------


class WassersteinRobustQR:
    """
    Distributionally robust quantile regression under p-Wasserstein ambiguity.

    For p = 2 (the recommended choice), the estimator solves:

        beta* = argmin_beta  (1/N) sum_i rho_tau(y_i - x_i^T beta - s_bar) +
                             c_{tau,2} * eps * ||beta||_2

        s* = s_bar* + (eps/2) * (tau^2 - (1-tau)^2) * c_{tau,2}^{-1} * ||beta*||_2

    Zhang, Mao & Wang (2026) prove this is the exact closed-form solution to
    the worst-case quantile regression problem over a 2-Wasserstein eps-ball
    around the empirical training distribution.

    Finite-sample guarantee (Theorem 3): set eps via ``optimal_eps()`` with
    moment parameter s > 2 and confidence 1 - eta. The estimator then achieves
    O(N^{-1/2}) out-of-sample risk with probability >= 1 - eta, dimension-free
    and without sub-Gaussian assumptions. Lognormal and Pareto-body severity
    distributions satisfy the s > 2 condition.

    For p = 1 (W_1 distance): the estimator is identical to standard QR
    regardless of eps. W_1 robustness is already implicit in the check loss.
    Setting p = 1 is a useful sanity check but gives no robustness benefit.

    Parameters
    ----------
    tau:
        Quantile level in (0, 1). Use tau >= 0.95 for large-loss applications.
    p:
        Wasserstein ball type. Must be an integer >= 1. Default 2 (recommended).
        p = 1 produces the same result as standard QR regardless of eps.
    eps:
        Wasserstein ball radius. If None and fit_eps is True, estimated from
        training data using Theorem 3 schedule with default s = 4.0, eta = 0.05.
        If None and fit_eps is False, defaults to 0 (standard QR).
    fit_eps:
        If True and eps is None, estimate eps automatically using the Theorem 3
        schedule. The resulting eps is stored in eps_used_ after fit().

    Attributes
    ----------
    coef_:
        Slope vector beta*, shape (n_features,). Set after fit().
    intercept_:
        Scalar intercept s* (includes Theorem 1 correction). Set after fit().
    eps_used_:
        Actual Wasserstein radius used. Set after fit().
    n_features_in_:
        Number of features seen during fit().
    is_fitted_:
        True after fit() has been called.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_quantile import WassersteinRobustQR
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(300, 2))
    >>> y = X[:, 0] + rng.exponential(size=300)
    >>> model = WassersteinRobustQR(tau=0.95, p=2, eps=0.1)
    >>> model.fit(X, y)
    WassersteinRobustQR(tau=0.95, p=2, eps=0.1)
    >>> preds = model.predict(X[:5])

    Notes
    -----
    The intercept correction is positive for tau > 0.5 and eps > 0, correcting
    the systematic downward bias of empirical high quantiles in small samples.
    For tau < 0.5, the correction is negative (upward-shifting the lower tail).
    For tau = 0.5, the correction is zero.

    References
    ----------
    Zhang, C., Mao, T., & Wang, R. (2026). Wasserstein Distributionally Robust
        Quantile Regression. arXiv:2603.14991.
    """

    def __init__(
        self,
        tau: float,
        p: int = 2,
        eps: float | None = None,
        fit_eps: bool = True,
    ) -> None:
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        if eps is not None and eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")

        self.tau = tau
        self.p = p
        self.eps = eps
        self.fit_eps = fit_eps

        # Set after fit()
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.eps_used_: float | None = None
        self.n_features_in_: int | None = None
        self.is_fitted_: bool = False

    def optimal_eps(
        self,
        N: int,
        s: float = 4.0,
        eta: float = 0.05,
    ) -> float:
        """
        Theorem 3 Wasserstein radius schedule.

        eps_N(eta) = c_{tau,p} * log(2N + 1)^(1/s) / sqrt(N)

        Under a finite s-th moment condition (s > 2), this radius guarantees
        that the WDRQR estimator achieves O(N^{-1/2}) out-of-sample risk with
        probability at least 1 - eta, dimension-free.

        Note: the confidence level eta appears implicitly through the choice of
        eps — the schedule is designed so that the worst-case risk bound holds
        with probability 1 - eta. The formula above is the Theorem 3 rate;
        for a given eta, you may scale eps by log(1/eta)^{1/2} for a tighter
        confidence-adapted bound.

        Parameters
        ----------
        N:
            Training sample size.
        s:
            Moment condition parameter. s > 2 is required; larger s means
            lighter-tailed distribution. For motor severity (lognormal-ish),
            s = 4 is reasonable. For heavy-tailed lines (Pareto-ish), use s = 2.5.
        eta:
            Nominal failure probability (unused in the base schedule but
            included for API consistency and future extensions).

        Returns
        -------
        Non-negative float. Decreases as O(N^{-1/2} log(N)^{1/s}).

        Raises
        ------
        ValueError
            If s <= 2 (moment condition violated) or N <= 0.
        """
        if s <= 2.0:
            raise ValueError(f"s must be > 2 for Theorem 3 to hold, got {s}")
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")
        c = _c_tau_p(self.tau, self.p)
        return c * (np.log(2.0 * N + 1.0) ** (1.0 / s)) / np.sqrt(N)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "WassersteinRobustQR":
        """
        Fit the robust quantile regression model.

        Parameters
        ----------
        X:
            Feature matrix of shape (N, d). Rows are observations, columns are
            features. Do not include an intercept column — the model fits its own.
        y:
            Response vector of shape (N,). Should be positive for loss severity
            applications.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X and y have incompatible shapes.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, got {X.shape[0]} and {y.shape[0]}"
            )

        N, d = X.shape
        self.n_features_in_ = d

        # Determine eps
        if self.eps is not None:
            eps = self.eps
        elif self.fit_eps:
            eps = self.optimal_eps(N)
        else:
            eps = 0.0

        if self.p == 1:
            # p=1: WDRQR is identical to standard QR (Zhang et al., §3.1).
            # eps is irrelevant; set to 0 for the solver.
            if eps > 0.0:
                warnings.warn(
                    "p=1: WDRQR under W_1 distance is identical to standard QR. "
                    "eps has no effect. Use p=2 for distributional robustness.",
                    UserWarning,
                    stacklevel=2,
                )
            eps_solve = 0.0
        else:
            eps_solve = eps

        beta_star, s_star = _fit_slope_and_intercept(X, y, self.tau, self.p, eps_solve)

        self.coef_ = beta_star
        self.intercept_ = s_star
        self.eps_used_ = eps
        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the tau-quantile for new observations.

        Parameters
        ----------
        X:
            Feature matrix of shape (M, d). Must have the same number of
            columns as the training data.

        Returns
        -------
        Predicted quantiles, shape (M,). For large tau (e.g. 0.99), these
        include the Theorem 1 intercept correction and will systematically
        exceed standard QR predictions when eps > 0.

        Raises
        ------
        RuntimeError
            If predict() is called before fit().
        ValueError
            If X has the wrong number of columns.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return X @ self.coef_ + self.intercept_

    def __repr__(self) -> str:
        return (
            f"WassersteinRobustQR(tau={self.tau}, p={self.p}, eps={self.eps})"
        )


# ---------------------------------------------------------------------------
# Convenience functions with Polars API
# ---------------------------------------------------------------------------


def wdrqr_large_loss_loading(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_score: pl.DataFrame,
    mean_model: Any,
    alpha: float = 0.99,
    eps: float | None = None,
) -> pl.Series:
    """
    Robust large loss loading: WDRQR quantile estimate minus mean model.

    The large loss loading is:

        loading_i = Q_{alpha}^{WDRQR}(x_i) - E[Y | x_i]

    where Q_{alpha}^{WDRQR} is the Wasserstein robust quantile and E[Y | x_i]
    comes from a separately fitted mean model (e.g. Tweedie GLM or GBM).

    This is a robust alternative to the TVaR-based loading in ``large_loss_loading``.
    It is most appropriate when the training segment is thin (N < 500) and there
    is genuine concern about distribution shift (inflation, regulatory change, etc.).

    Parameters
    ----------
    X_train:
        Training features as a Polars DataFrame. All columns are used as
        numeric features; do not include a target column.
    y_train:
        Training targets (loss amounts) as a Polars Series.
    X_score:
        Features for the risks to be scored. Columns must match X_train.
    mean_model:
        A fitted model with a predict(X) method. Accepts Polars DataFrames
        (converted to numpy float64 internally), numpy arrays, or any object
        whose predict() returns an array-like of mean predictions.
    alpha:
        Quantile level for the large loss loading. Default 0.99 (commercial
        lines standard). Use 0.95 for personal lines.
    eps:
        Wasserstein ball radius. If None, estimated automatically from the
        training sample size via Theorem 3.

    Returns
    -------
    Polars Series named "wdrqr_large_loss_loading" of per-risk loadings.
    Values may be negative if the robust quantile falls below the mean model
    prediction (possible for very low alpha or large mean model predictions).

    Notes
    -----
    The mean model should predict E[Y | X] on the same scale as y_train.
    If the mean model predicts log-scale or is fitted on severity only, adjust
    accordingly before calling this function.
    """
    X_train_np = X_train.to_numpy().astype(np.float64)
    y_train_np = y_train.to_numpy().astype(np.float64)
    X_score_np = X_score.to_numpy().astype(np.float64)

    model = WassersteinRobustQR(tau=alpha, p=2, eps=eps, fit_eps=(eps is None))
    model.fit(X_train_np, y_train_np)
    quantile_preds = model.predict(X_score_np)

    # Get mean model predictions, handling Polars and numpy outputs
    try:
        mean_preds = mean_model.predict(X_score)
    except (TypeError, AttributeError):
        mean_preds = mean_model.predict(X_score_np)

    if isinstance(mean_preds, pl.DataFrame):
        mean_vals = mean_preds.to_series().to_numpy().astype(np.float64)
    elif isinstance(mean_preds, pl.Series):
        mean_vals = mean_preds.to_numpy().astype(np.float64)
    else:
        mean_vals = np.asarray(mean_preds, dtype=np.float64)

    loading = quantile_preds - mean_vals
    return pl.Series("wdrqr_large_loss_loading", loading)


def wdrqr_reserve_quantile(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_score: pl.DataFrame,
    tau: float,
    eps: float | None = None,
    ci: bool = False,
) -> pl.DataFrame:
    """
    Per-risk robust reserve quantile with optional asymptotic confidence interval.

    Returns the WDRQR estimate of Q_tau(Y | X) for each row in X_score, with
    the Wasserstein radius eps determined from the training sample if not given.

    This function is designed for reserve capital quantile estimation in thin
    segments — for example, computing the 99.5th percentile for Solvency II
    capital allocation on a commercial liability segment with N = 150 claims.

    Parameters
    ----------
    X_train:
        Training features as a Polars DataFrame.
    y_train:
        Training targets (loss amounts) as a Polars Series.
    X_score:
        Features for the risks to be scored. Columns must match X_train.
    tau:
        Quantile level in (0, 1). Typical values: 0.95, 0.99, 0.995.
    eps:
        Wasserstein ball radius. If None, estimated via Theorem 3 schedule.
    ci:
        If True, include asymptotic +/- uncertainty columns derived from the
        eps radius. These are heuristic intervals indicating the range of
        quantile estimates across the Wasserstein ball, not formal coverage
        intervals. For rigorous coverage guarantees, use insurance-conformal
        (conformalized quantile regression) on top of these predictions.

    Returns
    -------
    Polars DataFrame with columns:
        - ``quantile``: robust quantile estimate Q_{tau}^{WDRQR}(x_i)
        - ``eps_used``: Wasserstein radius used (same for all rows)
        - ``quantile_lower`` (if ci=True): estimate with zero eps (standard QR)
        - ``quantile_upper`` (if ci=True): estimate with 2 * eps (wider ball)

    Notes
    -----
    The CI columns (quantile_lower, quantile_upper) are NOT prediction intervals
    in the frequentist sense. They reflect the sensitivity of the quantile
    estimate to the choice of eps — a useful diagnostic for how much the
    robustness loading matters for the given segment size. See the research
    report (arXiv:2603.14991, §4) for formal asymptotic theory.
    """
    X_train_np = X_train.to_numpy().astype(np.float64)
    y_train_np = y_train.to_numpy().astype(np.float64)
    X_score_np = X_score.to_numpy().astype(np.float64)

    model = WassersteinRobustQR(tau=tau, p=2, eps=eps, fit_eps=(eps is None))
    model.fit(X_train_np, y_train_np)
    quantile_preds = model.predict(X_score_np)
    eps_used = float(model.eps_used_)

    result: dict[str, Any] = {
        "quantile": quantile_preds,
        "eps_used": np.full(len(X_score_np), eps_used),
    }

    if ci:
        # Lower bound: standard QR (eps = 0)
        model_lower = WassersteinRobustQR(tau=tau, p=2, eps=0.0, fit_eps=False)
        model_lower.fit(X_train_np, y_train_np)
        result["quantile_lower"] = model_lower.predict(X_score_np)

        # Upper bound: twice the radius
        model_upper = WassersteinRobustQR(tau=tau, p=2, eps=2.0 * eps_used, fit_eps=False)
        model_upper.fit(X_train_np, y_train_np)
        result["quantile_upper"] = model_upper.predict(X_score_np)

    return pl.DataFrame(result)
