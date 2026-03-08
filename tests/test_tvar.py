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
        assert result.method == "grid_mean"

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
        The model has noise features so won't be exact; allow ±0.5.
        """
        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X, alpha=0.9)
        analytical_tvar = -np.log(0.1) + 1.0  # ≈ 3.303
        mean_tvar = float(result.values.mean())
        assert abs(mean_tvar - analytical_tvar) < 0.5, f"TVaR far off: {mean_tvar}"

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
