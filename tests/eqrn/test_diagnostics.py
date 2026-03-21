"""
Tests for EQRNDiagnostics (insurance_quantile.eqrn).

Smoke tests: verify plots produce without error and return Figure objects.
Numerical tests: verify summary_table produces finite values.
"""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt
from scipy.stats import genpareto

pytest.importorskip(
    "torch",
    reason="EQRN tests require torch. Install with: pip install insurance-quantile[eqrn]",
)

from insurance_quantile.eqrn import EQRNModel, EQRNDiagnostics


@pytest.fixture(scope="module")
def diag_setup():
    """Fitted model and test data for diagnostic tests."""
    rng = np.random.default_rng(200)
    n = 2500
    x1 = rng.uniform(0, 1, n)
    x2 = rng.standard_normal(n)
    X = np.column_stack([x1, x2])
    xi_true = 0.2 + 0.1 * x1
    sigma_true = 5000.0
    y = np.array([
        genpareto.rvs(c=xi_true[i], scale=sigma_true, loc=8000.0,
                      random_state=int(rng.integers(1e9)))
        for i in range(n)
    ])
    split = 2000
    model = EQRNModel(
        tau_0=0.75,
        hidden_sizes=(16, 8),
        shape_fixed=False,
        n_epochs=80,
        patience=20,
        seed=200,
        verbose=0,
    )
    model.fit(X[:split], y[:split])
    diag = EQRNDiagnostics(model)
    return diag, model, X[split:], y[split:], X, y


class TestQQPlot:
    def test_qq_plot_returns_figure(self, diag_setup):
        diag, model, X_test, y_test, X, y = diag_setup
        fig = diag.qq_plot(X_test, y_test)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_qq_plot_axes_labelled(self, diag_setup):
        diag, model, X_test, y_test, X, y = diag_setup
        fig = diag.qq_plot(X_test, y_test)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)


class TestCalibrationPlot:
    def test_calibration_returns_figure(self, diag_setup):
        diag, model, X_test, y_test, X, y = diag_setup
        fig = diag.calibration_plot(X_test, y_test)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_calibration_custom_levels(self, diag_setup):
        diag, model, X_test, y_test, X, y = diag_setup
        fig = diag.calibration_plot(X_test, y_test, levels=[0.85, 0.90, 0.95])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMeanResidualLifePlot:
    def test_mrl_returns_figure(self, diag_setup):
        diag, model, X_test, y_test, X, y = diag_setup
        fig = diag.mean_residual_life_plot(y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_mrl_custom_n_thresholds(self, diag_setup):
        diag, model, X_test, y_test, X, y = diag_setup
        fig = diag.mean_residual_life_plot(y, n_thresholds=20)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestXiScatter:
    def test_xi_scatter_returns_figure(self, diag_setup):
        diag, model, X_test, y_test, X, y = diag_setup
        fig = diag.xi_scatter(X_test, feat_idx=(0, 1))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_xi_scatter_with_feature_names(self, diag_setup):
        diag, model, X_test, y_test, X, y = diag_setup
        fig = diag.xi_scatter(X_test, feature_names=["x1", "x2"], feat_idx=(0, 1))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSummaryTable:
    def test_summary_table_returns_dataframe(self, diag_setup):
        import pandas as pd
        diag, model, X_test, y_test, X, y = diag_setup
        df = diag.summary_table(X_test, y_test)
        assert isinstance(df, pd.DataFrame)

    def test_summary_table_expected_columns(self, diag_setup):
        diag, model, X_test, y_test, X, y = diag_setup
        df = diag.summary_table(X_test, y_test)
        expected_cols = {
            "level",
            "predicted_exceedance_rate",
            "empirical_exceedance_rate",
            "mean_predicted_quantile",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_summary_table_finite_values(self, diag_setup):
        diag, model, X_test, y_test, X, y = diag_setup
        df = diag.summary_table(X_test, y_test)
        assert df["empirical_exceedance_rate"].notna().all()
        assert df["mean_predicted_quantile"].notna().all()

    def test_summary_table_exceedance_rates_reasonable(self, diag_setup):
        """Empirical exceedance rates should be non-negative."""
        diag, model, X_test, y_test, X, y = diag_setup
        df = diag.summary_table(X_test, y_test, levels=[0.85, 0.90, 0.95, 0.99])
        assert (df["empirical_exceedance_rate"] >= 0).all()
        assert (df["empirical_exceedance_rate"] <= 1).all()


class TestDiagnosticsUnfittedRaises:
    def test_unfitted_model_raises(self):
        """Creating diagnostics with an unfitted model raises ValueError."""
        model = EQRNModel()
        with pytest.raises(ValueError, match="fitted"):
            EQRNDiagnostics(model)
