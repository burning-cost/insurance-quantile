"""
Diagnostic tools for EQRN model assessment.

Provides visualisations and numerical summaries for evaluating how well
the fitted EQRN model describes the tail of the conditional distribution.

Four main diagnostics:
1. QQ plot — GPD probability integral transform on the exceedance set
2. Mean residual life plot — linearity above threshold confirms GPD
3. Threshold stability plot — xi and sigma* estimates vs tau_0 level
4. Calibration plot — empirical vs predicted exceedance probability
5. Xi map — scatter of xi(x) over the first two principal components

These follow standard EVT diagnostic practice (Coles 2001, Davison & Huser 2015).
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gpd import eqrn_exceedance_prob
from .intermediate import IntermediateQuantileEstimator
from .model import EQRNModel


class EQRNDiagnostics:
    """Diagnostic tools for an fitted EQRNModel.

    Parameters
    ----------
    model:
        A fitted EQRNModel instance.
    figsize:
        Default figure size for plots.
    """

    def __init__(
        self,
        model: EQRNModel,
        figsize: tuple[float, float] = (8.0, 5.5),
    ) -> None:
        if not model.is_fitted:
            raise ValueError("Model must be fitted before creating diagnostics.")
        self.model = model
        self.figsize = figsize

    def qq_plot(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ax: Optional[plt.Axes] = None,
        title: str = "GPD QQ plot (exceedances)",
    ) -> plt.Figure:
        """QQ plot of the probability integral transform on exceedances.

        If the GPD tail model is correctly specified, the transformed residuals
        should follow a standard Exponential(1) distribution. This plot shows
        observed vs expected quantiles on the exponential scale.

        Parameters
        ----------
        X:
            Covariate matrix for the assessment dataset.
        y:
            Response values.
        ax:
            Optional existing Axes to plot on.
        title:
            Plot title.

        Returns
        -------
        plt.Figure
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        xi, sigma, threshold = self.model._get_network_params(X)

        # Exceedances only
        exceed = y > threshold
        if exceed.sum() < 5:
            raise ValueError("Fewer than 5 exceedances; cannot produce QQ plot.")

        z_obs = y[exceed] - threshold[exceed]
        xi_e = xi[exceed]
        sigma_e = sigma[exceed]

        # PIT: F_GPD(z | xi, sigma) should be U[0,1]
        # 1 - S(z) = 1 - (1 + xi*z/sigma)^{-1/xi}
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(
                np.abs(xi_e) < 1e-8,
                1.0 - np.exp(-z_obs / sigma_e),
                1.0 - (1.0 + xi_e * z_obs / sigma_e) ** (-1.0 / xi_e),
            )
        u = np.clip(u, 0.0, 1.0)

        # Transform to Exp(1): -log(1 - u)
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        exp_residuals = -np.log(1.0 - u)

        # Theoretical Exp(1) quantiles
        n = len(exp_residuals)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical = -np.log(1.0 - probs)
        observed_sorted = np.sort(exp_residuals)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        ax.scatter(theoretical, observed_sorted, s=15, alpha=0.6, color="steelblue",
                   label=f"Exceedances (n={n})")
        max_val = max(theoretical.max(), observed_sorted.max())
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.2, label="y=x")
        ax.set_xlabel("Theoretical Exp(1) quantiles")
        ax.set_ylabel("Observed quantiles")
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def calibration_plot(
        self,
        X: np.ndarray,
        y: np.ndarray,
        levels: Sequence[float] = (0.85, 0.90, 0.95, 0.99, 0.995),
        ax: Optional[plt.Axes] = None,
        title: str = "Calibration: predicted vs empirical exceedance rate",
    ) -> plt.Figure:
        """Calibration plot comparing predicted and empirical exceedance rates.

        For each quantile level tau, plots predicted coverage (1 - tau) against
        the fraction of observations exceeding the predicted quantile. A
        well-calibrated model lies on the diagonal.

        Parameters
        ----------
        X:
            Covariate matrix.
        y:
            Response values.
        levels:
            Quantile levels to evaluate.
        ax:
            Optional Axes.
        title:
            Plot title.

        Returns
        -------
        plt.Figure
        """
        y = np.asarray(y, dtype=float)
        n = len(y)

        predicted_coverage = []
        empirical_coverage = []

        for tau in levels:
            if tau <= self.model.tau_0:
                continue
            q_pred = self.model.predict_quantile(X, q=tau)
            emp_rate = float((y > q_pred).mean())
            predicted_coverage.append(1.0 - tau)
            empirical_coverage.append(emp_rate)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        ax.scatter(predicted_coverage, empirical_coverage, s=60, zorder=5,
                   color="steelblue")
        for pc, ec, lv in zip(predicted_coverage, empirical_coverage, levels):
            ax.annotate(f"τ={lv}", (pc, ec), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)

        max_v = max(max(predicted_coverage), max(empirical_coverage)) * 1.2
        ax.plot([0, max_v], [0, max_v], "r--", linewidth=1.2, label="Perfect calibration")
        ax.set_xlabel("Predicted exceedance rate (1 - τ)")
        ax.set_ylabel("Empirical exceedance rate")
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def threshold_stability_plot(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau_range: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        title: str = "Threshold stability: shape parameter vs τ₀",
    ) -> plt.Figure:
        """Threshold stability analysis: mean xi_hat vs tau_0 level.

        Fits a shape_fixed=True EQRN at each tau_0 in tau_range and plots
        the fitted scalar xi against tau_0. Stability (flat region) indicates
        the threshold is in the GPD domain.

        Parameters
        ----------
        X:
            Covariate matrix.
        y:
            Response values.
        tau_range:
            Array of tau_0 levels to scan. Default linspace(0.7, 0.9, 9).
        seed:
            Random seed.
        ax:
            Optional Axes.
        title:
            Plot title.

        Returns
        -------
        plt.Figure
        """
        if tau_range is None:
            tau_range = np.linspace(0.7, 0.90, 9)

        xi_vals = []
        sigma_vals = []

        for tau_i in tau_range:
            mini_model = EQRNModel(
                tau_0=float(tau_i),
                hidden_sizes=(16, 8),
                shape_fixed=True,
                n_epochs=200,
                patience=30,
                seed=seed,
                verbose=0,
            )
            try:
                mini_model.fit(X, y)
                params = mini_model.predict_params(X)
                xi_vals.append(params["xi"].mean())
                sigma_vals.append(params["sigma"].mean())
            except Exception:
                xi_vals.append(np.nan)
                sigma_vals.append(np.nan)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        ax.plot(tau_range, xi_vals, "o-", color="steelblue", label="Mean ξ (shape)")
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Intermediate quantile level τ₀")
        ax.set_ylabel("Fitted shape parameter ξ")
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def mean_residual_life_plot(
        self,
        y: np.ndarray,
        n_thresholds: int = 50,
        ax: Optional[plt.Axes] = None,
        title: str = "Mean residual life plot",
    ) -> plt.Figure:
        """Mean residual life (mean excess) plot.

        Plots E[Y - u | Y > u] against u. For a GPD tail, this should be
        approximately linear in u. Linearity onset indicates where the GPD
        approximation becomes valid, guiding choice of tau_0.

        Parameters
        ----------
        y:
            Response values (all claims, not just exceedances).
        n_thresholds:
            Number of threshold values to evaluate.
        ax:
            Optional Axes.
        title:
            Plot title.

        Returns
        -------
        plt.Figure
        """
        y = np.asarray(y, dtype=float)
        y_sorted = np.sort(y)

        # Range from 10th to 95th percentile
        u_min = np.percentile(y, 10)
        u_max = np.percentile(y, 95)
        thresholds = np.linspace(u_min, u_max, n_thresholds)

        mean_excess = []
        lower_ci = []
        upper_ci = []

        for u in thresholds:
            excesses = y[y > u] - u
            if len(excesses) < 5:
                mean_excess.append(np.nan)
                lower_ci.append(np.nan)
                upper_ci.append(np.nan)
                continue
            me = excesses.mean()
            se = excesses.std() / np.sqrt(len(excesses))
            mean_excess.append(me)
            lower_ci.append(me - 1.96 * se)
            upper_ci.append(me + 1.96 * se)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        mean_excess = np.array(mean_excess)
        lower_ci = np.array(lower_ci)
        upper_ci = np.array(upper_ci)

        valid = ~np.isnan(mean_excess)
        ax.fill_between(thresholds[valid], lower_ci[valid], upper_ci[valid],
                        alpha=0.25, color="steelblue")
        ax.plot(thresholds[valid], mean_excess[valid], "o-", color="steelblue",
                markersize=3, label="Mean excess ± 1.96 SE")
        ax.set_xlabel("Threshold u")
        ax.set_ylabel("E[Y − u | Y > u]")
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def xi_scatter(
        self,
        X: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        feat_idx: tuple[int, int] = (0, 1),
        ax: Optional[plt.Axes] = None,
        title: str = "Shape parameter ξ(x) over covariate space",
    ) -> plt.Figure:
        """Scatter plot of xi(x) over two selected covariates.

        Parameters
        ----------
        X:
            Covariate matrix.
        feature_names:
            Optional feature names for axis labels.
        feat_idx:
            Indices of the two features to use as x and y axes.
        ax:
            Optional Axes.
        title:
            Plot title.

        Returns
        -------
        plt.Figure
        """
        params = self.model.predict_params(X)
        xi = params["xi"].values

        i, j = feat_idx
        x_feat = X[:, i]
        y_feat = X[:, j]

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        sc = ax.scatter(x_feat, y_feat, c=xi, cmap="RdYlBu_r", s=15, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="ξ(x)")

        xl = feature_names[i] if feature_names else f"Feature {i}"
        yl = feature_names[j] if feature_names else f"Feature {j}"
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def summary_table(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        levels: Sequence[float] = (0.90, 0.95, 0.99, 0.995),
    ) -> pd.DataFrame:
        """Summary table comparing predicted and empirical quantile coverage.

        Parameters
        ----------
        X_test:
            Test covariate matrix.
        y_test:
            Test response values.
        levels:
            Quantile levels to evaluate.

        Returns
        -------
        pd.DataFrame
            Columns: level, predicted_exceedance_rate, empirical_exceedance_rate,
            mean_predicted_quantile.
        """
        y_test = np.asarray(y_test, dtype=float)
        rows = []
        for tau in levels:
            if tau <= self.model.tau_0:
                continue
            q_pred = self.model.predict_quantile(X_test, q=tau)
            emp = float((y_test > q_pred).mean())
            rows.append({
                "level": tau,
                "predicted_exceedance_rate": round(1.0 - tau, 4),
                "empirical_exceedance_rate": round(emp, 4),
                "mean_predicted_quantile": round(q_pred.mean(), 2),
            })
        return pd.DataFrame(rows)
