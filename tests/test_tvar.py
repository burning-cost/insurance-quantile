"""
Tests for TVaR calculations: per_risk_tvar and portfolio_tvar.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_quantile import per_risk_tvar, portfolio_tvar
from insurance_quantile._types import TVaRResult


class TestPerRiskTVaR:
    def test_returns_tvar_result(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        assert isinstance(result, TVaRResult)

    def test_tvar_series_length(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        assert len(result.values) == len(X)

    def test_tvar_alpha_stored(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        assert result.alpha == 0.9

    def test_tvar_method_label(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        assert result.method == "trapezoidal"

    def test_tvar_exceeds_var(self, fitted_quantile_model, exponential_data):
        """TVaR should be >= VaR for all risks (TVaR is always in the tail)."""
        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        tvar = result.values.to_numpy()
        var = result.var_values.to_numpy()
        # Allow small numerical tolerance from isotonic regression
        assert np.all(tvar >= var - 0.05), "TVaR should be >= VaR per risk"

    def test_tvar_positive(self, fitted_quantile_model, exponential_data):
        """All TVaR values should be positive for positive-support distributions."""
        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        assert result.values.min() > 0

    def test_tvar_close_to_analytical(self, fitted_quantile_model, exponential_data):
        """
        For Exponential(1), TVaR_0.9 = Q(0.9) + 1 = -ln(0.1) + 1 ≈ 3.303.
        The model has noise features and limited iterations so won't be exact.
        CatBoost with 300 iterations on noise features may estimate high quantiles
        (Q(0.95), Q(0.99)) with significant variance. Allow ±1.5 to test the
        formula is reasonable without requiring model accuracy beyond the fixture.
        """
        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        analytical_tvar = -np.log(0.1) + 1.0  # ≈ 3.303
        mean_tvar = float(result.values.mean())
        assert abs(mean_tvar - analytical_tvar) < 1.5, f"TVaR far off: {mean_tvar}"

    def test_tvar_trapz_formula_correctness(self):
        """
        Verify that the trapezoidal integration formula produces < 5% relative error
        when given exact analytical quantile predictions.

        This test bypasses model fitting uncertainty by using a mock QuantileGBM
        whose predict() method returns the true Exponential(1) quantile function:
            Q(u) = -ln(1 - u)

        For Exponential(rate=1) with alpha=0.9:
            TVaR_0.9 = E[Y | Y > Q(0.9)] = Q(0.9) + 1 = -ln(0.1) + 1 ≈ 3.3026

        With exact quantile predictions at levels [0.9, 0.95, 0.975, 0.99, 0.999],
        the trapezoidal integral should recover TVaR to within 5% relative, since
        the quantile function of the exponential is smooth on this interval.

        Note: simple mean of quantile levels would give equal weight to Q(0.95)
        and Q(0.99) despite them spanning very different tail fractions (5% vs 4%
        of the distribution). The trapezoidal rule correctly weights by interval
        width, giving more accurate TVaR.
        """

        class _AnalyticalExponentialModel:
            """
            Mock QuantileGBM returning exact Exp(1) quantile predictions.
            Bypasses CatBoost fitting so the test isolates the integration formula.
            """

            class spec:
                quantiles = [0.9, 0.95, 0.975, 0.99, 0.999]
                column_names = ["q_0.9", "q_0.95", "q_0.975", "q_0.99", "q_0.999"]

            def predict(self, X: pl.DataFrame) -> pl.DataFrame:
                n = len(X)
                data = {}
                for q in self.spec.quantiles:
                    col = f"q_{q}"
                    # Exact Exp(1) quantile function: Q(u) = -ln(1-u)
                    val = -np.log(1.0 - q)
                    data[col] = [val] * n
                return pl.DataFrame(data)

        n = 10
        X_dummy = pl.DataFrame({"x": np.zeros(n)})
        model = _AnalyticalExponentialModel()

        result = per_risk_tvar(model, X_dummy, alpha=0.9)

        # Analytical TVaR_0.9 for Exp(1): Q(0.9) + 1/rate = -ln(0.1) + 1
        analytical_tvar = -np.log(0.1) + 1.0  # ≈ 3.3026
        mean_tvar = float(result.values.mean())

        rel_error = abs(mean_tvar - analytical_tvar) / analytical_tvar
        assert rel_error < 0.05, (
            f"Trapz TVaR formula has {rel_error:.1%} relative error vs analytical "
            f"({mean_tvar:.4f} vs {analytical_tvar:.4f}). "
            "Check that the integration uses np.trapz(q_values, quantile_levels) / (1-alpha), "
            "not a simple mean of quantile predictions."
        )

    def test_loading_over_var(self, fitted_quantile_model, exponential_data):
        """loading_over_var = TVaR - VaR should be non-negative."""
        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        loading = result.loading_over_var.to_numpy()
        assert np.all(loading >= -0.05)

    def test_alpha_invalid_zero(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        with pytest.raises(ValueError, match="alpha must be in"):
            per_risk_tvar(fitted_quantile_model, X, alpha=0.0)

    def test_alpha_invalid_one(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        with pytest.raises(ValueError, match="alpha must be in"):
            per_risk_tvar(fitted_quantile_model, X, alpha=1.0)

    def test_alpha_above_all_quantiles_raises(self, exponential_data):
        """If alpha is above all model quantiles, should raise ValueError."""
        from insurance_quantile import QuantileGBM
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.5, 0.7], iterations=100)
        model.fit(X, y)
        with pytest.raises(ValueError, match="no quantile levels above alpha"):
            per_risk_tvar(model, X, alpha=0.9)

    def test_different_alpha_levels(self, fitted_quantile_model, exponential_data):
        """Higher alpha should give higher TVaR (more extreme tail)."""
        X, _ = exponential_data
        result_80 = per_risk_tvar(fitted_quantile_model, X, alpha=0.75)
        result_90 = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        mean_80 = float(result_80.values.mean())
        mean_90 = float(result_90.values.mean())
        assert mean_90 > mean_80, "TVaR_0.9 should exceed TVaR_0.8"


class TestPortfolioTVaR:
    def test_returns_float(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        result = portfolio_tvar(fitted_quantile_model, X, alpha=0.9)
        assert isinstance(result, float)

    def test_mean_aggregation(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        per_risk = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        portfolio = portfolio_tvar(fitted_quantile_model, X, alpha=0.9, aggregate_method="mean")
        assert abs(portfolio - float(per_risk.values.mean())) < 1e-10

    def test_sum_aggregation(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        per_risk = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        portfolio = portfolio_tvar(fitted_quantile_model, X, alpha=0.9, aggregate_method="sum")
        assert abs(portfolio - float(per_risk.values.sum())) < 1e-6

    def test_invalid_aggregate_method(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        with pytest.raises(ValueError, match="aggregate_method"):
            portfolio_tvar(fitted_quantile_model, X, alpha=0.9, aggregate_method="median")

    def test_portfolio_positive(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        result = portfolio_tvar(fitted_quantile_model, X, alpha=0.9)
        assert result > 0

    def test_var_series_polars(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        assert isinstance(result.var_values, pl.Series)
        assert isinstance(result.values, pl.Series)
