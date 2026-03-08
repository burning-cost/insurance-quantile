"""
Tests for the types module: TailModel, TVaRResult, ExceedanceCurve.
"""

from __future__ import annotations

import polars as pl
import pytest

from insurance_quantile._types import ExceedanceCurve, QuantileSpec, TailModel, TVaRResult


class TestTailModel:
    def test_construction(self):
        spec = QuantileSpec(quantiles=[0.5, 0.9])
        meta = TailModel(
            spec=spec,
            n_features=5,
            feature_names=["a", "b", "c", "d", "e"],
            n_training_rows=1000,
        )
        assert meta.n_features == 5
        assert meta.n_training_rows == 1000
        assert meta.fix_crossing is True

    def test_catboost_params_default_empty(self):
        spec = QuantileSpec(quantiles=[0.5])
        meta = TailModel(spec=spec, n_features=2, feature_names=["x", "y"], n_training_rows=100)
        assert meta.catboost_params == {}

    def test_fix_crossing_default(self):
        spec = QuantileSpec(quantiles=[0.5])
        meta = TailModel(spec=spec, n_features=1, feature_names=["x"], n_training_rows=10)
        assert meta.fix_crossing is True


class TestTVaRResult:
    def test_loading_over_var(self):
        tvar_vals = pl.Series("tvar", [3.0, 4.0, 5.0])
        var_vals = pl.Series("var", [2.0, 2.5, 3.0])
        result = TVaRResult(alpha=0.9, values=tvar_vals, var_values=var_vals)
        loading = result.loading_over_var
        assert loading.to_list() == [1.0, 1.5, 2.0]

    def test_alpha_stored(self):
        result = TVaRResult(
            alpha=0.95,
            values=pl.Series("tvar", [1.0]),
            var_values=pl.Series("var", [0.5]),
        )
        assert result.alpha == 0.95

    def test_method_default(self):
        result = TVaRResult(
            alpha=0.9,
            values=pl.Series("tvar", [1.0]),
            var_values=pl.Series("var", [0.5]),
        )
        assert result.method == "grid_mean"


class TestExceedanceCurveType:
    def test_construction(self):
        curve = ExceedanceCurve(
            thresholds=[0.0, 1.0],
            probabilities=[1.0, 0.5],
            n_risks=200,
        )
        assert curve.n_risks == 200

    def test_as_dataframe_returns_polars(self):
        curve = ExceedanceCurve(
            thresholds=[0.0, 1.0, 2.0],
            probabilities=[0.9, 0.5, 0.1],
            n_risks=50,
        )
        df = curve.as_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"threshold", "exceedance_prob"}
