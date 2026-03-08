"""
Tests for exceedance_curve and oep_curve.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_quantile import exceedance_curve, oep_curve
from insurance_quantile._types import ExceedanceCurve


class TestExceedanceCurve:
    def test_returns_polars_dataframe(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(100)
        result = exceedance_curve(fitted_quantile_model, X_small)
        assert isinstance(result, pl.DataFrame)

    def test_columns_present(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(100)
        result = exceedance_curve(fitted_quantile_model, X_small)
        assert "threshold" in result.columns
        assert "exceedance_prob" in result.columns
        assert "n_risks" in result.columns

    def test_default_n_thresholds(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(100)
        result = exceedance_curve(fitted_quantile_model, X_small, n_thresholds=100)
        assert len(result) == 100  # linspace(0, max, 100) gives exactly 100 points

    def test_custom_thresholds(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(50)
        thresholds = [0.0, 1.0, 2.0, 5.0, 10.0]
        result = exceedance_curve(fitted_quantile_model, X_small, thresholds=thresholds)
        assert len(result) == 5

    def test_exceedance_decreasing(self, fitted_quantile_model, exponential_data):
        """Exceedance probability should be non-increasing with threshold."""
        X, _ = exponential_data
        X_small = X.head(200)
        result = exceedance_curve(fitted_quantile_model, X_small)
        probs = result["exceedance_prob"].to_numpy()
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1] - 1e-9, (
                f"Exceedance curve not monotone at index {i}: {probs[i]} < {probs[i+1]}"
            )

    def test_exceedance_at_zero_threshold(self, fitted_quantile_model, exponential_data):
        """At threshold=0, exceedance prob should be close to 1 for positive distributions."""
        X, _ = exponential_data
        X_small = X.head(100)
        thresholds = [0.0, 1.0, 5.0]
        result = exceedance_curve(fitted_quantile_model, X_small, thresholds=thresholds)
        p_at_zero = float(result.filter(pl.col("threshold") == 0.0)["exceedance_prob"][0])
        assert p_at_zero > 0.8, f"Expected P(Y>0)≈1 for exponential, got {p_at_zero}"

    def test_exceedance_in_0_1(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(100)
        result = exceedance_curve(fitted_quantile_model, X_small)
        probs = result["exceedance_prob"].to_numpy()
        assert np.all(probs >= -0.01)
        assert np.all(probs <= 1.01)

    def test_n_risks_column_value(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(150)
        result = exceedance_curve(fitted_quantile_model, X_small)
        assert result["n_risks"].unique().to_list() == [150]


class TestOEPCurve:
    def test_returns_exceedance_curve_type(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(100)
        result = oep_curve(fitted_quantile_model, X_small)
        assert isinstance(result, ExceedanceCurve)

    def test_as_dataframe(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(100)
        result = oep_curve(fitted_quantile_model, X_small)
        df = result.as_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert "threshold" in df.columns
        assert "exceedance_prob" in df.columns

    def test_n_risks_stored(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(77)
        result = oep_curve(fitted_quantile_model, X_small)
        assert result.n_risks == 77

    def test_thresholds_and_probs_length_match(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(100)
        result = oep_curve(fitted_quantile_model, X_small, n_thresholds=50)
        assert len(result.thresholds) == len(result.probabilities)

    def test_independence_oep_range(self, fitted_quantile_model, exponential_data):
        """Under independence, OEP should be in [0, 1]."""
        X, _ = exponential_data
        X_small = X.head(50)
        result = oep_curve(fitted_quantile_model, X_small, independence_assumption=True)
        probs = np.array(result.probabilities)
        assert np.all(probs >= -0.01)
        assert np.all(probs <= 1.01)

    def test_independence_vs_mean_exceedance(self, fitted_quantile_model, exponential_data):
        """
        For a large portfolio, the independence OEP should be close to 1 for low thresholds,
        while mean exceedance gives the average per-risk exceedance.
        """
        X, _ = exponential_data
        X_small = X.head(200)
        oep_indep = oep_curve(fitted_quantile_model, X_small, independence_assumption=True)
        oep_mean = oep_curve(fitted_quantile_model, X_small, independence_assumption=False)
        # Independence OEP at low threshold should be high (at least one risk exceeds)
        # Mean exceedance gives average per-risk probability
        indep_at_zero = oep_indep.probabilities[0]
        mean_at_zero = oep_mean.probabilities[0]
        # For a 200-risk portfolio, independence OEP at threshold=0 should be > mean exceedance
        assert indep_at_zero >= mean_at_zero - 0.1

    def test_custom_thresholds_oep(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(100)
        thresholds = [0.0, 0.5, 1.0, 2.0, 5.0]
        result = oep_curve(fitted_quantile_model, X_small, thresholds=thresholds)
        assert len(result.thresholds) == 5

    def test_oep_decreasing(self, fitted_quantile_model, exponential_data):
        """OEP should be non-increasing."""
        X, _ = exponential_data
        X_small = X.head(100)
        result = oep_curve(fitted_quantile_model, X_small)
        probs = result.probabilities
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1] - 1e-9


class TestExceedanceCurveDataclass:
    def test_as_dataframe_structure(self):
        curve = ExceedanceCurve(
            thresholds=[0.0, 1.0, 2.0],
            probabilities=[0.9, 0.5, 0.1],
            n_risks=100,
        )
        df = curve.as_dataframe()
        assert df.shape == (3, 2)
        assert df["threshold"].to_list() == [0.0, 1.0, 2.0]
        assert df["exceedance_prob"].to_list() == [0.9, 0.5, 0.1]
