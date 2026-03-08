"""
Tests for QuantileGBM: fit, predict, quantile/expectile modes,
isotonic crossing fix, and Polars I/O.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_quantile import QuantileGBM, QuantileSpec


# ---------------------------------------------------------------------------
# QuantileSpec validation
# ---------------------------------------------------------------------------

class TestQuantileSpec:
    def test_valid_spec(self):
        spec = QuantileSpec(quantiles=[0.1, 0.5, 0.9])
        assert spec.quantiles == [0.1, 0.5, 0.9]
        assert spec.mode == "quantile"

    def test_column_names(self):
        spec = QuantileSpec(quantiles=[0.5, 0.9, 0.99])
        assert spec.column_names == ["q_0.5", "q_0.9", "q_0.99"]

    def test_empty_quantiles_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            QuantileSpec(quantiles=[])

    def test_quantile_out_of_range_raises(self):
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            QuantileSpec(quantiles=[0.0, 0.5])

    def test_quantile_exactly_one_raises(self):
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            QuantileSpec(quantiles=[1.0])

    def test_non_monotone_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            QuantileSpec(quantiles=[0.9, 0.5])

    def test_duplicate_quantile_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            QuantileSpec(quantiles=[0.5, 0.5])

    def test_expectile_mode(self):
        spec = QuantileSpec(quantiles=[0.5, 0.9], mode="expectile")
        assert spec.mode == "expectile"


# ---------------------------------------------------------------------------
# QuantileGBM: basic construction
# ---------------------------------------------------------------------------

class TestQuantileGBMConstruction:
    def test_default_params(self):
        model = QuantileGBM(quantiles=[0.5, 0.9])
        assert model.spec.quantiles == [0.5, 0.9]
        assert not model.is_fitted

    def test_is_not_fitted_initially(self):
        model = QuantileGBM(quantiles=[0.5])
        assert not model.is_fitted

    def test_metadata_before_fit_raises(self):
        model = QuantileGBM(quantiles=[0.5])
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.metadata

    def test_predict_before_fit_raises(self):
        model = QuantileGBM(quantiles=[0.5])
        X = pl.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)


# ---------------------------------------------------------------------------
# QuantileGBM: fit and predict (quantile mode)
# ---------------------------------------------------------------------------

class TestQuantileGBMFitPredict:
    def test_fit_returns_self(self, exponential_data):
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.5, 0.9], iterations=100)
        result = model.fit(X, y)
        assert result is model

    def test_is_fitted_after_fit(self, exponential_data):
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.5, 0.9], iterations=100)
        model.fit(X, y)
        assert model.is_fitted

    def test_predict_returns_polars_dataframe(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        preds = fitted_quantile_model.predict(X)
        assert isinstance(preds, pl.DataFrame)

    def test_predict_column_names(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        preds = fitted_quantile_model.predict(X)
        expected = ["q_0.1", "q_0.25", "q_0.5", "q_0.75", "q_0.9", "q_0.95", "q_0.99"]
        assert preds.columns == expected

    def test_predict_output_shape(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        preds = fitted_quantile_model.predict(X)
        assert preds.shape == (len(X), 7)

    def test_predictions_non_negative(self, fitted_quantile_model, exponential_data):
        """Exponential losses should have non-negative quantile predictions."""
        X, _ = exponential_data
        preds = fitted_quantile_model.predict(X)
        for col in preds.columns:
            assert preds[col].min() >= -0.5, f"Column {col} has large negative values"

    def test_quantile_monotonicity(self, fitted_quantile_model, exponential_data):
        """After isotonic fix, each row must have non-decreasing quantile predictions."""
        X, _ = exponential_data
        preds = fitted_quantile_model.predict(X)
        cols = preds.columns
        for i in range(len(cols) - 1):
            diff = preds[cols[i + 1]] - preds[cols[i]]
            assert diff.min() >= -1e-9, f"Crossing between {cols[i]} and {cols[i+1]}"

    def test_median_close_to_analytical(self, fitted_quantile_model, exponential_data):
        """Exponential(1) median = ln(2) ≈ 0.693. Model should be close."""
        X, _ = exponential_data
        preds = fitted_quantile_model.predict(X)
        median_mean = preds["q_0.5"].mean()
        assert abs(median_mean - np.log(2)) < 0.15, f"Median far off: {median_mean}"

    def test_p90_close_to_analytical(self, fitted_quantile_model, exponential_data):
        """Exponential(1) Q(0.9) = -ln(0.1) ≈ 2.303."""
        X, _ = exponential_data
        preds = fitted_quantile_model.predict(X)
        p90_mean = preds["q_0.9"].mean()
        assert abs(p90_mean - (-np.log(0.1))) < 0.3, f"P90 far off: {p90_mean}"

    def test_metadata_populated(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        meta = fitted_quantile_model.metadata
        assert meta.n_features == 3
        assert meta.feature_names == ["x0", "x1", "x2"]
        assert meta.n_training_rows == 2000

    def test_single_quantile_fit(self, exponential_data):
        """Single quantile should work without isotonic regression issues."""
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.9], iterations=100, fix_crossing=True)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X), 1)
        assert preds.columns == ["q_0.9"]


# ---------------------------------------------------------------------------
# QuantileGBM: expectile mode
# ---------------------------------------------------------------------------

class TestExpectileMode:
    def test_expectile_fit_and_predict(self, exponential_data):
        X, y = exponential_data
        model = QuantileGBM(
            quantiles=[0.5, 0.9],
            use_expectile=True,
            iterations=150,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X), 2)
        assert preds.columns == ["q_0.5", "q_0.9"]

    def test_expectile_is_fitted(self, fitted_expectile_model):
        assert fitted_expectile_model.is_fitted

    def test_expectile_column_names(self, fitted_expectile_model, exponential_data):
        X, _ = exponential_data
        preds = fitted_expectile_model.predict(X)
        assert preds.columns == ["q_0.5", "q_0.75", "q_0.9", "q_0.95"]

    def test_expectile_mode_in_spec(self, fitted_expectile_model):
        assert fitted_expectile_model.spec.mode == "expectile"

    def test_expectile_predictions_non_negative(self, fitted_expectile_model, exponential_data):
        X, _ = exponential_data
        preds = fitted_expectile_model.predict(X)
        for col in preds.columns:
            assert preds[col].min() >= -0.5

    def test_expectile_monotonicity(self, fitted_expectile_model, exponential_data):
        """Higher expectile level should give higher prediction (after isotonic fix)."""
        X, _ = exponential_data
        preds = fitted_expectile_model.predict(X)
        cols = preds.columns
        for i in range(len(cols) - 1):
            diff = preds[cols[i + 1]] - preds[cols[i]]
            assert diff.min() >= -1e-9


# ---------------------------------------------------------------------------
# Isotonic crossing fix
# ---------------------------------------------------------------------------

class TestIsotonicCrossing:
    def test_crossing_fix_enabled_by_default(self, exponential_data):
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.1, 0.5, 0.9], iterations=100)
        assert model._fix_crossing is True

    def test_no_crossing_with_fix(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        preds = fitted_quantile_model.predict(X)
        arr = preds.to_numpy()
        for row in arr:
            for i in range(len(row) - 1):
                assert row[i] <= row[i + 1] + 1e-9

    def test_fix_crossing_false(self, exponential_data):
        """Model with fix_crossing=False should still run but may have crossings."""
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.25, 0.75], fix_crossing=False, iterations=100)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X), 2)


# ---------------------------------------------------------------------------
# Exposure weighting
# ---------------------------------------------------------------------------

class TestExposureWeighting:
    def test_fit_with_exposure(self, exponential_data):
        X, y = exponential_data
        exposure = pl.Series("w", np.ones(len(y)))
        model = QuantileGBM(quantiles=[0.5, 0.9], iterations=100)
        model.fit(X, y, exposure=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X), 2)

    def test_exposure_affects_fit(self, exponential_data):
        """
        Downweighting high-loss rows should shift quantile predictions down.
        This is a directional test: we don't require a specific magnitude.
        """
        rng = np.random.default_rng(1)
        X, y = exponential_data
        y_np = y.to_numpy()

        # Upweight low-loss rows; downweight high-loss rows
        high_loss_mask = y_np > np.quantile(y_np, 0.9)
        w = np.where(high_loss_mask, 0.1, 1.0)
        exposure = pl.Series("w", w)

        m_uniform = QuantileGBM(quantiles=[0.9], iterations=200, depth=4)
        m_uniform.fit(X, y)

        m_weighted = QuantileGBM(quantiles=[0.9], iterations=200, depth=4)
        m_weighted.fit(X, y, exposure=exposure)

        p_uniform = m_uniform.predict(X)["q_0.9"].mean()
        p_weighted = m_weighted.predict(X)["q_0.9"].mean()
        assert p_weighted < p_uniform, "Downweighting large losses should reduce q90"

    def test_fit_with_none_exposure(self, exponential_data):
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.5], iterations=100)
        model.fit(X, y, exposure=None)
        assert model.is_fitted


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_row_predict(self, fitted_quantile_model):
        X_single = pl.DataFrame({"x0": [0.0], "x1": [0.0], "x2": [0.0]})
        preds = fitted_quantile_model.predict(X_single)
        assert preds.shape[0] == 1

    def test_extreme_low_alpha(self, exponential_data):
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.01, 0.99], iterations=100)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X), 2)
        assert (preds["q_0.99"] - preds["q_0.01"]).min() >= -1e-9

    def test_high_quantile_count(self, exponential_data):
        """Fit with many quantile levels should still produce correct column names."""
        X, y = exponential_data
        qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        model = QuantileGBM(quantiles=qs, iterations=100)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.columns == [f"q_{q}" for q in qs]

    def test_polars_output_types(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        preds = fitted_quantile_model.predict(X)
        for col in preds.columns:
            assert preds[col].dtype == pl.Float64

    def test_catboost_kwargs_passed(self, exponential_data):
        """iterations and depth should be respected."""
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.5], iterations=10, depth=2)
        model.fit(X, y)
        assert model.is_fitted


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

class TestCalibrationReport:
    def test_calibration_report_structure(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        report = fitted_quantile_model.calibration_report(X, y)
        assert "coverage" in report
        assert "pinball_loss" in report
        assert "mean_pinball_loss" in report

    def test_calibration_report_keys(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        report = fitted_quantile_model.calibration_report(X, y)
        expected_keys = ["q_0.1", "q_0.25", "q_0.5", "q_0.75", "q_0.9", "q_0.95", "q_0.99"]
        assert list(report["coverage"].keys()) == expected_keys

    def test_coverage_values_in_range(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        report = fitted_quantile_model.calibration_report(X, y)
        for q_str, cov in report["coverage"].items():
            assert 0.0 <= cov <= 1.0, f"Coverage out of range for {q_str}: {cov}"

    def test_pinball_loss_non_negative(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        report = fitted_quantile_model.calibration_report(X, y)
        for q_str, loss in report["pinball_loss"].items():
            assert loss >= 0.0, f"Negative pinball loss for {q_str}"

    def test_mean_pinball_loss_is_average(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        report = fitted_quantile_model.calibration_report(X, y)
        individual = list(report["pinball_loss"].values())
        expected_mean = np.mean(individual)
        assert abs(report["mean_pinball_loss"] - expected_mean) < 1e-10
