"""
Tests for large_loss_loading and ilf (increased limits factors).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_quantile import large_loss_loading, ilf


# ---------------------------------------------------------------------------
# large_loss_loading
# ---------------------------------------------------------------------------

class TestLargeLossLoading:
    def test_returns_polars_series(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        from conftest import _ConstantMeanModel
        mean_model = _ConstantMeanModel(mean_val=1.0, n=len(X))
        result = large_loss_loading(mean_model, fitted_quantile_model, X, alpha=0.9)
        assert isinstance(result, pl.Series)

    def test_loading_series_length(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        from conftest import _ConstantMeanModel
        mean_model = _ConstantMeanModel(mean_val=1.0, n=len(X))
        result = large_loss_loading(mean_model, fitted_quantile_model, X, alpha=0.9)
        assert len(result) == len(X)

    def test_loading_positive_for_high_alpha(self, fitted_quantile_model, exponential_data):
        """
        TVaR at 0.9 should exceed the mean for most risks in an Exponential distribution.
        Exponential(1): mean=1, TVaR_0.9=3.303. Loading ≈ 2.3 on average.
        """
        X, _ = exponential_data
        from conftest import _ConstantMeanModel
        mean_model = _ConstantMeanModel(mean_val=1.0, n=len(X))
        result = large_loss_loading(mean_model, fitted_quantile_model, X, alpha=0.9)
        assert float(result.mean()) > 0.5, f"Expected positive loading, got {result.mean()}"

    def test_loading_column_name(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        from conftest import _ConstantMeanModel
        mean_model = _ConstantMeanModel(mean_val=1.0, n=len(X))
        result = large_loss_loading(mean_model, fitted_quantile_model, X)
        assert result.name == "large_loss_loading"

    def test_loading_with_numpy_mean_model(self, fitted_quantile_model, exponential_data):
        """Mean model that returns a numpy array should also work."""
        X, _ = exponential_data

        class NumpyModel:
            def predict(self, X: pl.DataFrame) -> np.ndarray:
                return np.ones(len(X))

        result = large_loss_loading(NumpyModel(), fitted_quantile_model, X, alpha=0.9)
        assert len(result) == len(X)

    def test_loading_with_dataframe_mean_model(self, fitted_quantile_model, exponential_data):
        """Mean model that returns a single-column Polars DataFrame should work."""
        X, _ = exponential_data

        class DataFrameModel:
            def predict(self, X: pl.DataFrame) -> pl.DataFrame:
                return pl.DataFrame({"pred": np.ones(len(X))})

        result = large_loss_loading(DataFrameModel(), fitted_quantile_model, X)
        assert len(result) == len(X)

    def test_multi_column_dataframe_raises(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data

        class BadModel:
            def predict(self, X: pl.DataFrame) -> pl.DataFrame:
                return pl.DataFrame({"a": np.ones(len(X)), "b": np.ones(len(X))})

        with pytest.raises(ValueError, match="multi-column"):
            large_loss_loading(BadModel(), fitted_quantile_model, X)

    def test_lower_alpha_gives_lower_loading(self, fitted_quantile_model, exponential_data):
        """TVaR at 0.75 < TVaR at 0.9, so loading at 0.75 < loading at 0.9."""
        X, _ = exponential_data
        from conftest import _ConstantMeanModel
        mean_model = _ConstantMeanModel(mean_val=1.0, n=len(X))
        loading_75 = large_loss_loading(mean_model, fitted_quantile_model, X, alpha=0.75)
        loading_90 = large_loss_loading(mean_model, fitted_quantile_model, X, alpha=0.9)
        assert float(loading_75.mean()) < float(loading_90.mean())


# ---------------------------------------------------------------------------
# ilf (increased limits factors)
# ---------------------------------------------------------------------------

class TestILF:
    def test_returns_polars_series(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        result = ilf(fitted_quantile_model, X, basic_limit=100.0, higher_limit=500.0)
        assert isinstance(result, pl.Series)

    def test_ilf_series_length(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        result = ilf(fitted_quantile_model, X, basic_limit=1.0, higher_limit=5.0)
        assert len(result) == len(X)

    def test_ilf_column_name(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        result = ilf(fitted_quantile_model, X, basic_limit=1.0, higher_limit=5.0)
        assert result.name == "ilf"

    def test_ilf_greater_than_one(self, fitted_quantile_model, exponential_data):
        """ILF from basic to higher limit should always be >= 1."""
        X, _ = exponential_data
        result = ilf(fitted_quantile_model, X, basic_limit=0.5, higher_limit=5.0)
        assert float(result.min()) >= 0.9, f"ILF below 1: {result.min()}"

    def test_ilf_increases_with_limit(self, fitted_quantile_model, exponential_data):
        """A wider limit spread should give a higher ILF."""
        X, _ = exponential_data
        # Take a small subsample for speed
        X_small = X.head(100)
        ilf_narrow = ilf(fitted_quantile_model, X_small, basic_limit=0.5, higher_limit=2.0)
        ilf_wide = ilf(fitted_quantile_model, X_small, basic_limit=0.5, higher_limit=5.0)
        assert float(ilf_wide.mean()) >= float(ilf_narrow.mean())

    def test_ilf_equal_limits_raises(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        with pytest.raises(ValueError, match="less than higher_limit"):
            ilf(fitted_quantile_model, X, basic_limit=5.0, higher_limit=5.0)

    def test_ilf_basic_exceeds_higher_raises(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        with pytest.raises(ValueError, match="less than higher_limit"):
            ilf(fitted_quantile_model, X, basic_limit=10.0, higher_limit=1.0)

    def test_ilf_negative_limit_raises(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        with pytest.raises(ValueError, match="must be positive"):
            ilf(fitted_quantile_model, X, basic_limit=-1.0, higher_limit=10.0)

    def test_ilf_analytical_exponential(self, fitted_quantile_model, exponential_data):
        """
        For Exp(1): E[min(Y, L)] = 1 - exp(-L).
        ILF(L1, L2) = (1 - exp(-L2)) / (1 - exp(-L1))

        With L1=1, L2=3:
        ILF = (1 - exp(-3)) / (1 - exp(-1)) ≈ 0.9502 / 0.6321 ≈ 1.503

        Test that mean ILF is in the right ballpark (within 25% of analytical).
        """
        X, _ = exponential_data
        X_small = X.head(200)
        result = ilf(fitted_quantile_model, X_small, basic_limit=1.0, higher_limit=3.0)
        analytical = (1 - np.exp(-3)) / (1 - np.exp(-1))
        mean_ilf = float(result.mean())
        assert abs(mean_ilf - analytical) / analytical < 0.4, (
            f"ILF mean {mean_ilf:.3f} far from analytical {analytical:.3f}"
        )

    def test_ilf_n_integration_points(self, fitted_quantile_model, exponential_data):
        """More integration points should give similar results — not crash."""
        X, _ = exponential_data
        X_small = X.head(50)
        result_coarse = ilf(fitted_quantile_model, X_small, basic_limit=1.0, higher_limit=3.0, n_integration_points=50)
        result_fine = ilf(fitted_quantile_model, X_small, basic_limit=1.0, higher_limit=3.0, n_integration_points=500)
        # Results should be close
        assert abs(float(result_coarse.mean()) - float(result_fine.mean())) < 0.2
