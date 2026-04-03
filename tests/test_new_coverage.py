"""
Extended test coverage for insurance-quantile.

Covers:
    - _es_regressor internal helpers (_pseudo_values, _auto_bin_count,
      _make_bin_edges, _assign_bins, _local_es_estimate, _check_loss_scalar)
    - _model.py additional methods: calibration_report, predict_tvar,
      metadata property, is_fitted property
    - _tvar.py edge cases: alpha exactly at a fitted quantile level,
      alpha not at a fitted quantile (flat extrapolation path), TVaR >= VaR
    - _types.py: TwoPartResult field types, ExceedanceCurve.as_dataframe,
      QuantileSpec properties
    - _loading.py: MeanModelWrapper, ilf validation and mathematical properties,
      large_loss_loading with multi-column DataFrame raise
    - _two_part.py: gamma=0, gamma=1, all-fallback scenario, negative loading,
      no mean_sev_model (trapezoid approximation path), numpy input type error
    - _calibration.py: single-observation edge case, extreme alpha values
    - _exceedance.py: independence assumption, n_risks stored
    - Mathematical properties across modules:
        ES predictions >= sample VaR
        TVaR >= VaR (coherence)
        Quantile monotonicity from QuantileGBM
        Pinball loss is a proper scoring rule

All tests use small synthetic data. CatBoost model fixtures from conftest.py
are reused where needed (session-scoped, fit once).
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Helpers and small model builders
# ---------------------------------------------------------------------------


def _make_np_data(n: int = 200, p: int = 2, seed: int = 42):
    """Return (X_np, y_np) arrays for numpy-based tests."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, p))
    y = rng.exponential(size=n)
    return X, y


def _make_pl_data(n: int = 200, p: int = 2, seed: int = 42):
    """Return (X_pl, y_pl) Polars objects."""
    X_np, y_np = _make_np_data(n, p, seed)
    X = pl.DataFrame({f"x{i}": X_np[:, i] for i in range(p)})
    y = pl.Series("y", y_np)
    return X, y


# ===========================================================================
# _es_regressor internals
# ===========================================================================


class TestPseudoValues:
    """_pseudo_values: E[Z_i] = ES(alpha, x_i) when q_hat is exact."""

    def test_pseudo_value_shape(self):
        from insurance_quantile._es_regressor import _pseudo_values

        rng = np.random.default_rng(0)
        y = rng.exponential(size=100)
        q = np.quantile(y, 0.9) * np.ones(100)
        Z = _pseudo_values(y, q, alpha=0.9)
        assert Z.shape == (100,)

    def test_pseudo_value_lower_bound(self):
        """Z_i >= q_hat_i because excess is non-negative and division is by (1-alpha) > 0."""
        from insurance_quantile._es_regressor import _pseudo_values

        rng = np.random.default_rng(1)
        y = rng.exponential(size=200)
        q = np.quantile(y, 0.8) * np.ones(200)
        Z = _pseudo_values(y, q, alpha=0.8)
        # Z_i = q_hat + (y - q)_+ / (1-alpha) >= q_hat since (y - q)_+ >= 0
        assert np.all(Z >= q - 1e-12)

    def test_pseudo_value_mean_approximates_es(self):
        """
        For a large iid sample with constant q_hat = true Q(alpha):
            E[Z_i] ≈ ES(alpha) for Exp(1) at alpha=0.9.
        ES(0.9) for Exp(1) = -ln(0.1) + 1 ≈ 3.303.
        """
        from insurance_quantile._es_regressor import _pseudo_values

        rng = np.random.default_rng(2)
        n = 10000
        y = rng.exponential(size=n)
        alpha = 0.9
        q_hat = np.full(n, np.quantile(y, alpha))
        Z = _pseudo_values(y, q_hat, alpha)
        true_es = -np.log(0.1) + 1.0
        assert abs(Z.mean() - true_es) < 0.3, (
            f"Pseudo-value mean {Z.mean():.3f} far from true ES {true_es:.3f}"
        )

    def test_pseudo_values_equal_q_when_all_below(self):
        """When all y < q_hat, (y - q)_+ = 0, so Z_i = q_hat_i."""
        from insurance_quantile._es_regressor import _pseudo_values

        y = np.array([0.1, 0.2, 0.3])
        q = np.array([1.0, 1.0, 1.0])
        Z = _pseudo_values(y, q, alpha=0.9)
        np.testing.assert_array_equal(Z, q)


class TestAutoBinCount:
    def test_returns_int(self):
        from insurance_quantile._es_regressor import _auto_bin_count

        k = _auto_bin_count(200, 2)
        assert isinstance(k, int)

    def test_at_least_2(self):
        from insurance_quantile._es_regressor import _auto_bin_count

        # Edge cases: tiny n or p=0 should still return >= 2
        assert _auto_bin_count(1, 1) >= 2
        assert _auto_bin_count(10, 1) >= 2
        assert _auto_bin_count(0, 2) >= 2

    def test_increases_with_n(self):
        """More data -> more bins (generally)."""
        from insurance_quantile._es_regressor import _auto_bin_count

        k_small = _auto_bin_count(100, 1)
        k_large = _auto_bin_count(10000, 1)
        assert k_large >= k_small

    def test_single_dimension_formula(self):
        """p=1: k = ceil(1.6 * sqrt(n) / log(n)) for large n."""
        from insurance_quantile._es_regressor import _auto_bin_count

        k = _auto_bin_count(1000, 1)
        expected = int(np.ceil(1.6 * np.sqrt(1000) / np.log(1000)))
        assert k == max(expected, 2)


class TestMakeBinEdges:
    def test_returns_list_of_length_p(self):
        from insurance_quantile._es_regressor import _make_bin_edges

        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(100, 3))
        edges = _make_bin_edges(X, k=4)
        assert len(edges) == 3

    def test_each_edge_is_array(self):
        from insurance_quantile._es_regressor import _make_bin_edges

        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, size=(100, 2))
        edges = _make_bin_edges(X, k=3)
        for e in edges:
            assert isinstance(e, np.ndarray)

    def test_bin_edges_within_data_range(self):
        from insurance_quantile._es_regressor import _make_bin_edges

        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, size=(200, 2))
        edges = _make_bin_edges(X, k=5)
        for j, e in enumerate(edges):
            if len(e) > 0:
                assert np.all(e >= X[:, j].min() - 1e-10)
                assert np.all(e <= X[:, j].max() + 1e-10)

    def test_degenerate_constant_column_no_crash(self):
        """A constant column should not crash."""
        from insurance_quantile._es_regressor import _make_bin_edges

        X = np.ones((50, 2))
        X[:, 1] = np.random.default_rng(3).uniform(0, 1, 50)
        edges = _make_bin_edges(X, k=4)
        assert len(edges) == 2


class TestAssignBins:
    def test_returns_integer_array(self):
        from insurance_quantile._es_regressor import _assign_bins, _make_bin_edges

        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(100, 2))
        edges = _make_bin_edges(X, k=4)
        bins = _assign_bins(X, edges)
        assert bins.shape == (100,)

    def test_all_observations_assigned(self):
        from insurance_quantile._es_regressor import _assign_bins, _make_bin_edges

        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, size=(50, 2))
        edges = _make_bin_edges(X, k=3)
        bins = _assign_bins(X, edges)
        assert np.all(bins >= 0)
        assert len(bins) == 50


class TestLocalESEstimate:
    def test_returns_float_for_sufficient_obs(self):
        from insurance_quantile._es_regressor import _local_es_estimate

        rng = np.random.default_rng(0)
        X_bin = rng.uniform(0, 1, size=(20, 2))
        Z_bin = rng.exponential(size=20)
        result = _local_es_estimate(X_bin, Z_bin)
        assert isinstance(result, float)

    def test_returns_none_for_too_few_obs(self):
        """With m < p+1, OLS is underdetermined — should return None."""
        from insurance_quantile._es_regressor import _local_es_estimate

        X_bin = np.ones((2, 3))  # m=2, p=3: need at least 4 rows
        Z_bin = np.ones(2)
        result = _local_es_estimate(X_bin, Z_bin)
        assert result is None

    def test_recovers_constant_intercept(self):
        """
        When Z_bin = constant, OLS intercept should be close to that constant.
        """
        from insurance_quantile._es_regressor import _local_es_estimate

        rng = np.random.default_rng(42)
        X_bin = rng.uniform(0, 1, size=(50, 2))
        Z_bin = np.full(50, 5.0)
        result = _local_es_estimate(X_bin, Z_bin)
        assert result is not None
        assert abs(result - 5.0) < 0.5


class TestCheckLossScalar:
    def test_non_negative(self):
        """Check loss is always non-negative."""
        from insurance_quantile._es_regressor import _check_loss_scalar

        rng = np.random.default_rng(7)
        residuals = rng.normal(0, 1, 100)
        for alpha in [0.1, 0.5, 0.9]:
            loss = _check_loss_scalar(residuals, alpha)
            assert loss >= 0.0

    def test_asymmetry_high_alpha(self):
        """For alpha=0.9, underprediction penalised 9x more than overprediction."""
        from insurance_quantile._es_regressor import _check_loss_scalar

        res_under = np.array([1.0])   # positive residual: y > q
        res_over = np.array([-1.0])   # negative residual: y < q
        alpha = 0.9
        loss_under = _check_loss_scalar(res_under, alpha)
        loss_over = _check_loss_scalar(res_over, alpha)
        assert abs(loss_under - 0.9) < 1e-10
        assert abs(loss_over - 0.1) < 1e-10

    def test_symmetric_at_median(self):
        """At alpha=0.5, over and underprediction cost equally."""
        from insurance_quantile._es_regressor import _check_loss_scalar

        loss_under = _check_loss_scalar(np.array([1.0]), 0.5)
        loss_over = _check_loss_scalar(np.array([-1.0]), 0.5)
        assert abs(loss_under - loss_over) < 1e-10


# ===========================================================================
# _model.py: calibration_report, predict_tvar, metadata, is_fitted
# ===========================================================================


class TestQuantileGBMMetadata:
    def test_is_fitted_after_fit(self, fitted_quantile_model):
        assert fitted_quantile_model.is_fitted is True

    def test_is_fitted_before_fit(self):
        from insurance_quantile import QuantileGBM

        m = QuantileGBM(quantiles=[0.5, 0.9])
        assert m.is_fitted is False

    def test_metadata_raises_before_fit(self):
        from insurance_quantile import QuantileGBM

        m = QuantileGBM(quantiles=[0.5, 0.9])
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.metadata

    def test_metadata_after_fit(self, fitted_quantile_model):
        from insurance_quantile._types import TailModel

        meta = fitted_quantile_model.metadata
        assert isinstance(meta, TailModel)
        assert meta.n_features > 0
        assert meta.n_training_rows > 0

    def test_metadata_feature_names(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        meta = fitted_quantile_model.metadata
        assert len(meta.feature_names) == X.width

    def test_predict_before_fit_raises(self):
        from insurance_quantile import QuantileGBM

        m = QuantileGBM(quantiles=[0.5, 0.9])
        X = pl.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict(X)


class TestQuantileGBMCalibrationReport:
    def test_returns_dict(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        X_small = X.head(200)
        y_small = y.head(200)
        report = fitted_quantile_model.calibration_report(X_small, y_small)
        assert isinstance(report, dict)

    def test_report_keys(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        X_small = X.head(200)
        y_small = y.head(200)
        report = fitted_quantile_model.calibration_report(X_small, y_small)
        assert "coverage" in report
        assert "pinball_loss" in report
        assert "mean_pinball_loss" in report

    def test_coverage_keys_match_quantiles(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        X_small = X.head(200)
        y_small = y.head(200)
        report = fitted_quantile_model.calibration_report(X_small, y_small)
        expected_cols = fitted_quantile_model.spec.column_names
        assert set(report["coverage"].keys()) == set(expected_cols)

    def test_coverage_in_range(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        X_small = X.head(200)
        y_small = y.head(200)
        report = fitted_quantile_model.calibration_report(X_small, y_small)
        for v in report["coverage"].values():
            assert 0.0 <= v <= 1.0

    def test_mean_pinball_loss_positive(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        X_small = X.head(200)
        y_small = y.head(200)
        report = fitted_quantile_model.calibration_report(X_small, y_small)
        assert report["mean_pinball_loss"] > 0.0


class TestQuantileGBMPredictTVaR:
    def test_predict_tvar_returns_polars_series(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        tvar = fitted_quantile_model.predict_tvar(X.head(100), alpha=0.9)
        assert isinstance(tvar, pl.Series)

    def test_predict_tvar_length(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        X_small = X.head(50)
        tvar = fitted_quantile_model.predict_tvar(X_small, alpha=0.9)
        assert len(tvar) == 50

    def test_predict_tvar_positive(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        tvar = fitted_quantile_model.predict_tvar(X.head(100), alpha=0.9)
        assert float(tvar.min()) > 0.0

    def test_predict_tvar_exceeds_quantile(self, fitted_quantile_model, exponential_data):
        """TVaR(0.9) should exceed Q(0.9) per risk (with small tolerance for numerics)."""
        X, _ = exponential_data
        X_small = X.head(100)
        tvar = fitted_quantile_model.predict_tvar(X_small, alpha=0.9)
        preds = fitted_quantile_model.predict(X_small)
        var = preds["q_0.9"].to_numpy()
        tvar_np = tvar.to_numpy()
        assert np.all(tvar_np >= var - 0.05)

    def test_predict_tvar_no_quantiles_above_raises(self, exponential_data):
        from insurance_quantile import QuantileGBM

        X, y = exponential_data
        m = QuantileGBM(quantiles=[0.5, 0.7], iterations=50)
        m.fit(X.head(200), y.head(200))
        with pytest.raises(ValueError, match="no quantile levels above"):
            m.predict_tvar(X.head(10), alpha=0.95)

    def test_predict_tvar_alpha_zero_raises(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        with pytest.raises(ValueError, match="alpha must be in"):
            fitted_quantile_model.predict_tvar(X.head(10), alpha=0.0)

    def test_predict_tvar_alpha_above_all_quantiles_raises(self, exponential_data):
        """alpha > max_quantile should raise ValueError."""
        from insurance_quantile import QuantileGBM

        X, y = exponential_data
        m = QuantileGBM(quantiles=[0.5, 0.9], iterations=50)
        m.fit(X.head(200), y.head(200))
        # alpha=0.95 > max quantile 0.9 -> no quantiles above, raises
        with pytest.raises(ValueError, match="no quantile levels above"):
            m.predict_tvar(X.head(10), alpha=0.95)

    def test_predict_tvar_before_fit_raises(self):
        from insurance_quantile import QuantileGBM

        m = QuantileGBM(quantiles=[0.5, 0.9])
        X = pl.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict_tvar(X, alpha=0.5)

    def test_predict_tvar_series_name(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        tvar = fitted_quantile_model.predict_tvar(X.head(10), alpha=0.9)
        assert tvar.name == "tvar"


# ===========================================================================
# _tvar.py: edge cases
# ===========================================================================


class TestTVaREdgeCases:
    def test_alpha_exactly_at_fitted_quantile(self, fitted_quantile_model, exponential_data):
        """
        When alpha = 0.9 and model has q_0.9, the boundary value should come from
        Q(0.9) itself. This exercises the 'at_alpha' branch in per_risk_tvar.
        The fitted_quantile_model has quantiles [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].
        """
        from insurance_quantile import per_risk_tvar

        X, _ = exponential_data
        X_small = X.head(100)
        result = per_risk_tvar(fitted_quantile_model, X_small, alpha=0.9)
        tvar = result.values.to_numpy()
        preds = fitted_quantile_model.predict(X_small)
        var = preds["q_0.9"].to_numpy()
        # TVaR must be >= VaR
        assert np.all(tvar >= var - 1e-6)

    def test_alpha_not_at_fitted_quantile(self, fitted_quantile_model, exponential_data):
        """
        When alpha=0.85 is NOT in model quantiles, the boundary falls back to
        Q(first_above=0.9). TVaR should still be positive.
        """
        from insurance_quantile import per_risk_tvar

        X, _ = exponential_data
        X_small = X.head(100)
        result = per_risk_tvar(fitted_quantile_model, X_small, alpha=0.85)
        tvar = result.values.to_numpy()
        assert float(tvar.min()) > 0.0

    def test_alpha_above_all_quantiles_raises(self, exponential_data):
        """alpha above max quantile raises ValueError (no quantiles above alpha)."""
        from insurance_quantile import QuantileGBM, per_risk_tvar

        X, y = exponential_data
        m = QuantileGBM(quantiles=[0.9, 0.95], iterations=50)
        m.fit(X.head(200), y.head(200))
        # alpha = 0.99 > max_quantile = 0.95 -> no above quantiles -> ValueError
        with pytest.raises(ValueError, match="no quantile levels above"):
            per_risk_tvar(m, X.head(10), alpha=0.99)

    def test_tvar_higher_for_higher_alpha(self, fitted_quantile_model, exponential_data):
        """TVaR(0.95) > TVaR(0.9) in expectation (more extreme tail)."""
        from insurance_quantile import per_risk_tvar

        X, _ = exponential_data
        X_small = X.head(200)
        r90 = per_risk_tvar(fitted_quantile_model, X_small, alpha=0.9)
        r95 = per_risk_tvar(fitted_quantile_model, X_small, alpha=0.95)
        assert float(r95.values.mean()) >= float(r90.values.mean()) - 0.1

    def test_loading_over_var_non_negative(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import per_risk_tvar

        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X.head(200), alpha=0.9)
        loading = result.loading_over_var.to_numpy()
        assert np.all(loading >= -1e-6)

    def test_tvar_result_alpha_stored(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import per_risk_tvar

        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X.head(50), alpha=0.95)
        assert result.alpha == 0.95

    def test_tvar_method_is_trapezoidal(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import per_risk_tvar

        X, _ = exponential_data
        result = per_risk_tvar(fitted_quantile_model, X.head(50), alpha=0.9)
        assert result.method == "trapezoidal"


# ===========================================================================
# _types.py: extended
# ===========================================================================


class TestQuantileSpecExtended:
    def test_column_names_format(self):
        from insurance_quantile._types import QuantileSpec

        spec = QuantileSpec(quantiles=[0.1, 0.5, 0.9, 0.99])
        assert spec.column_names == ["q_0.1", "q_0.5", "q_0.9", "q_0.99"]

    def test_mode_expectile(self):
        from insurance_quantile._types import QuantileSpec

        spec = QuantileSpec(quantiles=[0.5, 0.9], mode="expectile")
        assert spec.mode == "expectile"

    def test_single_quantile_valid(self):
        from insurance_quantile._types import QuantileSpec

        spec = QuantileSpec(quantiles=[0.5])
        assert spec.quantiles == [0.5]

    def test_extreme_quantiles_valid(self):
        """Quantiles at 0.001 and 0.999 are valid."""
        from insurance_quantile._types import QuantileSpec

        spec = QuantileSpec(quantiles=[0.001, 0.999])
        assert len(spec.quantiles) == 2

    def test_quantile_zero_invalid(self):
        from insurance_quantile._types import QuantileSpec

        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            QuantileSpec(quantiles=[0.0, 0.5])

    def test_quantile_one_invalid(self):
        from insurance_quantile._types import QuantileSpec

        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            QuantileSpec(quantiles=[0.5, 1.0])


class TestExceedanceCurveAsDataFrame:
    def test_columns(self):
        from insurance_quantile._types import ExceedanceCurve

        curve = ExceedanceCurve(
            thresholds=[0.0, 1.0, 2.0],
            probabilities=[0.9, 0.5, 0.1],
            n_risks=100,
        )
        df = curve.as_dataframe()
        assert set(df.columns) == {"threshold", "exceedance_prob"}

    def test_row_count(self):
        from insurance_quantile._types import ExceedanceCurve

        curve = ExceedanceCurve(
            thresholds=list(range(5)),
            probabilities=[1.0, 0.8, 0.6, 0.3, 0.1],
            n_risks=50,
        )
        df = curve.as_dataframe()
        assert len(df) == 5

    def test_returns_polars_dataframe(self):
        from insurance_quantile._types import ExceedanceCurve

        curve = ExceedanceCurve(thresholds=[0.0], probabilities=[1.0], n_risks=10)
        assert isinstance(curve.as_dataframe(), pl.DataFrame)


class TestTVaRResultType:
    def test_loading_over_var_series(self):
        from insurance_quantile._types import TVaRResult

        r = TVaRResult(
            alpha=0.9,
            values=pl.Series("tvar", [3.0, 4.0]),
            var_values=pl.Series("var", [2.0, 2.5]),
        )
        loading = r.loading_over_var
        assert isinstance(loading, pl.Series)
        assert loading.to_list() == [1.0, 1.5]

    def test_method_trapezoidal(self):
        from insurance_quantile._types import TVaRResult

        r = TVaRResult(
            alpha=0.9,
            values=pl.Series("tvar", [3.0]),
            var_values=pl.Series("var", [2.0]),
            method="trapezoidal",
        )
        assert r.method == "trapezoidal"

    def test_default_method(self):
        from insurance_quantile._types import TVaRResult

        r = TVaRResult(
            alpha=0.9,
            values=pl.Series("tvar", [3.0]),
            var_values=pl.Series("var", [2.0]),
        )
        # Default method in TVaRResult is "grid_mean"
        assert r.method == "grid_mean"


# ===========================================================================
# _loading.py: MeanModelWrapper, ILF validation, large_loss_loading edge cases
# ===========================================================================


class TestMeanModelWrapper:
    def test_predict_returns_polars_series(self):
        from insurance_quantile import MeanModelWrapper

        class SimpleModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.ones(len(X))

        wrapper = MeanModelWrapper(SimpleModel())
        X = pl.DataFrame({"x0": [1.0, 2.0, 3.0], "x1": [0.5, 0.5, 0.5]})
        result = wrapper.predict(X)
        assert isinstance(result, pl.Series)
        assert result.name == "mean"

    def test_predict_length(self):
        from insurance_quantile import MeanModelWrapper

        class ConstantModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.full(len(X), 2.0)

        wrapper = MeanModelWrapper(ConstantModel())
        X = pl.DataFrame({"x": [1.0] * 10})
        result = wrapper.predict(X)
        assert len(result) == 10

    def test_predict_values_correct(self):
        from insurance_quantile import MeanModelWrapper

        class IdentityModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return X[:, 0]

        wrapper = MeanModelWrapper(IdentityModel())
        X = pl.DataFrame({"x0": [1.0, 2.0, 3.0]})
        result = wrapper.predict(X)
        np.testing.assert_allclose(result.to_numpy(), [1.0, 2.0, 3.0])

    def test_passes_numpy_to_wrapped_model(self):
        """The wrapped model receives a numpy array, not a Polars DataFrame."""
        from insurance_quantile import MeanModelWrapper

        received_type = []

        class TypeCapture:
            def predict(self, X):
                received_type.append(type(X).__name__)
                return np.ones(len(X))

        wrapper = MeanModelWrapper(TypeCapture())
        X = pl.DataFrame({"x0": [1.0, 2.0], "x1": [3.0, 4.0]})
        wrapper.predict(X)
        assert received_type[0] == "ndarray"


class TestILFValidation:
    def test_basic_limit_zero_raises(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import ilf

        X, _ = exponential_data
        with pytest.raises(ValueError, match="positive"):
            ilf(fitted_quantile_model, X.head(10), basic_limit=0, higher_limit=1000)

    def test_higher_limit_negative_raises(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import ilf

        X, _ = exponential_data
        with pytest.raises(ValueError, match="positive"):
            ilf(fitted_quantile_model, X.head(10), basic_limit=100, higher_limit=-1)

    def test_basic_limit_ge_higher_limit_raises(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import ilf

        X, _ = exponential_data
        with pytest.raises(ValueError, match="basic_limit"):
            ilf(fitted_quantile_model, X.head(10), basic_limit=500, higher_limit=500)

    def test_basic_limit_greater_than_higher_raises(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import ilf

        X, _ = exponential_data
        with pytest.raises(ValueError, match="basic_limit"):
            ilf(fitted_quantile_model, X.head(10), basic_limit=1000, higher_limit=100)

    def test_ilf_returns_polars_series(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import ilf

        X, _ = exponential_data
        result = ilf(fitted_quantile_model, X.head(50), basic_limit=1.0, higher_limit=5.0)
        assert isinstance(result, pl.Series)

    def test_ilf_series_length(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import ilf

        X, _ = exponential_data
        result = ilf(fitted_quantile_model, X.head(30), basic_limit=1.0, higher_limit=10.0)
        assert len(result) == 30

    def test_ilf_column_name(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import ilf

        X, _ = exponential_data
        result = ilf(fitted_quantile_model, X.head(10), basic_limit=1.0, higher_limit=5.0)
        assert result.name == "ilf"

    def test_ilf_at_least_one(self, fitted_quantile_model, exponential_data):
        """
        ILF(L1, L2) = E[min(Y,L2)] / E[min(Y,L1)] >= 1 since L2 > L1 implies
        min(Y,L2) >= min(Y,L1). Allow small numerical tolerance from integration.
        """
        from insurance_quantile import ilf

        X, _ = exponential_data
        result = ilf(fitted_quantile_model, X.head(50), basic_limit=1.0, higher_limit=5.0)
        ilf_vals = result.to_numpy()
        # Allow for numerical edge cases (integration approximation)
        assert np.all(ilf_vals >= 1.0 - 1e-3), f"ILF below 1.0: {ilf_vals.min():.4f}"


class TestLargeLossLoadingEdgeCases:
    def test_multi_column_dataframe_raises(self, fitted_quantile_model, exponential_data):
        """A mean model returning a multi-column DataFrame should raise ValueError."""
        from insurance_quantile import large_loss_loading

        X, _ = exponential_data
        X_small = X.head(50)

        class BadModel:
            def predict(self, X: pl.DataFrame) -> pl.DataFrame:
                return pl.DataFrame({"a": np.ones(len(X)), "b": np.ones(len(X))})

        with pytest.raises(ValueError, match="multi-column"):
            large_loss_loading(BadModel(), fitted_quantile_model, X_small, alpha=0.9)

    def test_numpy_fallback_mean_model(self, fitted_quantile_model, exponential_data):
        """A model that raises TypeError on Polars input falls back to numpy."""
        from insurance_quantile import large_loss_loading

        X, _ = exponential_data
        X_small = X.head(50)

        class NumpyOnlyModel:
            def predict(self, X):
                if isinstance(X, pl.DataFrame):
                    raise TypeError("does not accept Polars")
                return np.ones(len(X))

        result = large_loss_loading(NumpyOnlyModel(), fitted_quantile_model, X_small)
        assert len(result) == 50

    def test_series_name_is_large_loss_loading(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import large_loss_loading

        X, _ = exponential_data
        X_small = X.head(30)

        class ConstMean:
            def predict(self, X):
                return pl.Series("mean", np.ones(len(X)))

        result = large_loss_loading(ConstMean(), fitted_quantile_model, X_small)
        assert result.name == "large_loss_loading"


# ===========================================================================
# _calibration.py: edge cases
# ===========================================================================


class TestCalibrationEdgeCases:
    def test_pinball_loss_single_observation(self):
        """Single observation should not raise."""
        from insurance_quantile import pinball_loss

        y = pl.Series("y", [1.0])
        q = pl.Series("q", [2.0])
        loss = pinball_loss(y, q, alpha=0.9)
        assert loss >= 0.0

    def test_pinball_loss_extreme_alpha_low(self):
        """alpha=0.001: underprediction penalised very lightly."""
        from insurance_quantile import pinball_loss

        y = pl.Series("y", [2.0])
        q = pl.Series("q", [1.0])
        loss = pinball_loss(y, q, alpha=0.001)
        assert abs(loss - 0.001 * 1.0) < 1e-10

    def test_pinball_loss_extreme_alpha_high(self):
        """alpha=0.999: underprediction penalised very heavily."""
        from insurance_quantile import pinball_loss

        y = pl.Series("y", [2.0])
        q = pl.Series("q", [1.0])
        loss = pinball_loss(y, q, alpha=0.999)
        assert abs(loss - 0.999 * 1.0) < 1e-10

    def test_coverage_check_single_quantile(self, fitted_quantile_model, exponential_data):
        """coverage_check should work with a single quantile level."""
        from insurance_quantile import coverage_check

        X, y = exponential_data
        preds = fitted_quantile_model.predict(X.head(200))
        result = coverage_check(y.head(200), preds, [0.5])
        assert len(result) == 1

    def test_coverage_check_all_above_threshold(self):
        """If all y << q, coverage should be 1.0."""
        from insurance_quantile import coverage_check

        y = pl.Series("y", [0.1, 0.2, 0.3])
        preds = pl.DataFrame({"q_0.9": [100.0, 100.0, 100.0]})
        result = coverage_check(y, preds, [0.9])
        assert float(result["observed_coverage"][0]) == 1.0

    def test_coverage_check_all_below_threshold(self):
        """If all y >> q, coverage should be 0.0."""
        from insurance_quantile import coverage_check

        y = pl.Series("y", [10.0, 20.0, 30.0])
        preds = pl.DataFrame({"q_0.5": [0.1, 0.1, 0.1]})
        result = coverage_check(y, preds, [0.5])
        assert float(result["observed_coverage"][0]) == 0.0

    def test_coverage_error_column_equals_observed_minus_expected(self):
        """coverage_error = observed_coverage - expected_coverage."""
        from insurance_quantile import coverage_check

        rng = np.random.default_rng(5)
        n = 500
        y = pl.Series("y", rng.exponential(size=n))
        q = -np.log(1 - 0.5)  # true Exp(1) median
        preds = pl.DataFrame({"q_0.5": np.full(n, q)})
        result = coverage_check(y, preds, [0.5])
        obs = float(result["observed_coverage"][0])
        exp = float(result["expected_coverage"][0])
        err = float(result["coverage_error"][0])
        assert abs(err - (obs - exp)) < 1e-10


# ===========================================================================
# _exceedance.py: independence assumption, n_risks stored
# ===========================================================================


class TestOEPCurveWithIndependence:
    def test_independence_true_returns_exceedance_curve_type(
        self, fitted_quantile_model, exponential_data
    ):
        from insurance_quantile import oep_curve
        from insurance_quantile._types import ExceedanceCurve

        X, _ = exponential_data
        X_small = X.head(50)
        curve = oep_curve(
            fitted_quantile_model, X_small, n_thresholds=20,
            independence_assumption=True,
        )
        assert isinstance(curve, ExceedanceCurve)

    def test_independence_false_emits_warning(
        self, fitted_quantile_model, exponential_data
    ):
        from insurance_quantile import oep_curve

        X, _ = exponential_data
        X_small = X.head(50)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            oep_curve(fitted_quantile_model, X_small, n_thresholds=10,
                      independence_assumption=False)
        assert any(
            "mean" in str(warning.message).lower() or
            "not the portfolio" in str(warning.message).lower()
            for warning in w
        )

    def test_independence_curve_probabilities_in_range(
        self, fitted_quantile_model, exponential_data
    ):
        from insurance_quantile import oep_curve

        X, _ = exponential_data
        X_small = X.head(30)
        curve = oep_curve(
            fitted_quantile_model, X_small, n_thresholds=20,
            independence_assumption=True,
        )
        probs = curve.probabilities
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_oep_n_risks_stored(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import oep_curve

        X, _ = exponential_data
        X_small = X.head(40)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            curve = oep_curve(fitted_quantile_model, X_small, n_thresholds=5,
                              independence_assumption=False)
        assert curve.n_risks == 40

    def test_exceedance_curve_n_risks_column(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import exceedance_curve

        X, _ = exponential_data
        X_small = X.head(60)
        result = exceedance_curve(fitted_quantile_model, X_small, n_thresholds=10)
        n_risks_col = result["n_risks"].to_numpy()
        assert np.all(n_risks_col == 60)


# ===========================================================================
# _two_part.py: edge cases not in test_two_part.py
# ===========================================================================

class _MockFreqModel:
    classes_ = np.array([0, 1])

    def __init__(self, p_no_claim):
        self._p = np.asarray(p_no_claim, dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        p = self._p if len(self._p) == n else np.full(n, self._p[0])
        return np.column_stack([p, 1.0 - p])


class _MockMeanSevModel:
    def __init__(self, mean_sev: float):
        self._mean = mean_sev

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), float(self._mean))


def _make_mock_sev(quantiles, q_vals, n):
    """Build a QuantileGBM-like object returning fixed quantile values."""
    from insurance_quantile import QuantileGBM
    from insurance_quantile._types import TailModel

    model = QuantileGBM(quantiles=quantiles, iterations=1)
    q_df = pl.DataFrame(
        {f"q_{q}": np.full(n, q_vals[i]) for i, q in enumerate(quantiles)}
    )

    class _Predict:
        def __call__(self, X):
            return q_df

    model._is_fitted = True
    model.predict = _Predict()
    model._metadata = TailModel(
        spec=model._spec,
        n_features=1,
        feature_names=["x"],
        n_training_rows=n,
    )
    return model


def _test_X(n=4):
    return pl.DataFrame({"x": np.ones(n)})


class TestTwoPartEdgeCases:
    def test_gamma_zero_premium_equals_pure_premium(self):
        """At gamma=0, premium must equal pure_premium for all valid policies."""
        from insurance_quantile import TwoPartQuantilePremium

        n = 4
        sev = _make_mock_sev([0.5, 0.7, 0.9], [500.0, 700.0, 900.0], n)
        freq = _MockFreqModel(np.full(n, 0.5))
        mean_sev = _MockMeanSevModel(600.0)
        tpqp = TwoPartQuantilePremium(freq, sev, mean_sev)
        result = tpqp.predict_premium(_test_X(n), tau=0.8, gamma=0.0)

        prem = result.premium.to_numpy()
        pure = result.pure_premium.to_numpy()
        np.testing.assert_allclose(prem, pure, atol=1e-9)

    def test_gamma_one_premium_equals_severity_quantile(self):
        """At gamma=1, premium = severity_quantile for valid policies."""
        from insurance_quantile import TwoPartQuantilePremium

        n = 4
        sev = _make_mock_sev([0.5, 0.7, 0.9], [500.0, 700.0, 900.0], n)
        freq = _MockFreqModel(np.full(n, 0.5))
        mean_sev = _MockMeanSevModel(600.0)
        tpqp = TwoPartQuantilePremium(freq, sev, mean_sev)
        result = tpqp.predict_premium(_test_X(n), tau=0.8, gamma=1.0)

        prem = result.premium.to_numpy()
        sev_q = result.severity_quantile.to_numpy()
        np.testing.assert_allclose(prem, sev_q, atol=1e-9)

    def test_safety_loading_zero_for_all_fallback_policies(self):
        """All-fallback policies (p_i >= tau) get zero safety loading."""
        from insurance_quantile import TwoPartQuantilePremium

        n = 4
        sev = _make_mock_sev([0.5, 0.7, 0.9], [500.0, 700.0, 900.0], n)
        freq = _MockFreqModel(np.full(n, 0.95))  # p_i=0.95 > tau=0.90
        tpqp = TwoPartQuantilePremium(freq, sev)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            result = tpqp.predict_premium(_test_X(n), tau=0.90, gamma=0.5)

        loading = result.safety_loading.to_numpy()
        np.testing.assert_allclose(loading, 0.0, atol=1e-9)
        assert result.n_fallback == n

    def test_fallback_warning_emitted(self):
        """UserWarning should be emitted when any policy has p_i >= tau."""
        from insurance_quantile import TwoPartQuantilePremium

        n = 4
        sev = _make_mock_sev([0.5, 0.7], [500.0, 700.0], n)
        freq = _MockFreqModel(np.full(n, 0.95))
        tpqp = TwoPartQuantilePremium(freq, sev)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tpqp.predict_premium(_test_X(n), tau=0.90, gamma=0.5)
        assert any("fallback" in str(warning.message).lower() for warning in w)

    def test_numpy_input_raises_type_error(self):
        """Passing a numpy array instead of Polars DataFrame should raise TypeError."""
        from insurance_quantile import TwoPartQuantilePremium

        n = 2
        sev = _make_mock_sev([0.5, 0.7], [500.0, 700.0], n)
        freq = _MockFreqModel(np.full(n, 0.5))
        tpqp = TwoPartQuantilePremium(freq, sev)
        X_np = np.ones((n, 1))
        with pytest.raises(TypeError, match="Polars DataFrame"):
            tpqp.predict_premium(X_np, tau=0.8, gamma=0.5)

    def test_adjusted_tau_within_range_for_valid_policies(self):
        """All valid adjusted_tau values should be in (0, 1)."""
        from insurance_quantile import TwoPartQuantilePremium

        n = 10
        sev = _make_mock_sev([0.4, 0.5, 0.6, 0.7, 0.8], [400.0, 500.0, 600.0, 700.0, 800.0], n)
        p_vals = np.linspace(0.1, 0.7, n)  # all < tau=0.9
        freq = _MockFreqModel(p_vals)
        tpqp = TwoPartQuantilePremium(freq, sev)
        result = tpqp.predict_premium(_test_X(n), tau=0.9, gamma=0.5)

        adj_tau = result.adjusted_tau.drop_nulls().to_numpy()
        assert np.all(adj_tau > 0.0)
        assert np.all(adj_tau < 1.0)

    def test_no_mean_sev_model_uses_trapezoid(self):
        """When mean_sev_model is None, mean severity from trapezoid; pure_premium > 0."""
        from insurance_quantile import TwoPartQuantilePremium

        n = 4
        sev = _make_mock_sev([0.3, 0.5, 0.7, 0.9], [300.0, 500.0, 700.0, 900.0], n)
        freq = _MockFreqModel(np.full(n, 0.5))
        tpqp = TwoPartQuantilePremium(freq, sev, mean_sev_model=None)
        result = tpqp.predict_premium(_test_X(n), tau=0.8, gamma=0.5)
        assert float(result.pure_premium.min()) > 0.0

    def test_premium_field_names(self):
        """Result Series carry the documented field names."""
        from insurance_quantile import TwoPartQuantilePremium

        n = 3
        sev = _make_mock_sev([0.5, 0.7, 0.9], [500.0, 700.0, 900.0], n)
        freq = _MockFreqModel(np.full(n, 0.6))
        tpqp = TwoPartQuantilePremium(freq, sev)
        result = tpqp.predict_premium(_test_X(n), tau=0.9, gamma=0.5)
        assert result.premium.name == "premium"
        assert result.pure_premium.name == "pure_premium"
        assert result.safety_loading.name == "safety_loading"
        assert result.no_claim_prob.name == "no_claim_prob"

    def test_pure_premium_formula(self):
        """pure_premium = (1 - p_i) * E[S~_i | x_i]."""
        from insurance_quantile import TwoPartQuantilePremium

        n = 4
        sev = _make_mock_sev([0.5, 0.7, 0.9], [500.0, 700.0, 900.0], n)
        p_i = 0.6
        freq = _MockFreqModel(np.full(n, p_i))
        mean_sev = _MockMeanSevModel(700.0)
        tpqp = TwoPartQuantilePremium(freq, sev, mean_sev)
        result = tpqp.predict_premium(_test_X(n), tau=0.9, gamma=0.0)
        # At gamma=0, premium = pure_premium = (1 - p_i) * mean_sev
        expected_pure = (1.0 - p_i) * 700.0
        np.testing.assert_allclose(
            result.pure_premium.to_numpy(),
            expected_pure,
            atol=1e-6,
        )


# ===========================================================================
# _robust.py: additional edge cases
# ===========================================================================


class TestWDRQRAdditional:
    def test_predict_with_1d_input_raises(self):
        """predict() on 1D input should raise (2D required)."""
        from insurance_quantile import WassersteinRobustQR

        X, y = _make_np_data(100, 2)
        m = WassersteinRobustQR(tau=0.9, p=2, eps=0.0, fit_eps=False)
        m.fit(X, y)
        with pytest.raises(ValueError, match="2-dimensional"):
            m.predict(X[:5, 0])  # 1D

    def test_coef_is_numpy_array(self):
        from insurance_quantile import WassersteinRobustQR

        X, y = _make_np_data(100, 3)
        m = WassersteinRobustQR(tau=0.9, p=2, eps=0.0, fit_eps=False)
        m.fit(X, y)
        assert isinstance(m.coef_, np.ndarray)
        assert m.coef_.shape == (3,)

    def test_eps_used_is_float(self):
        from insurance_quantile import WassersteinRobustQR

        X, y = _make_np_data(100, 2)
        m = WassersteinRobustQR(tau=0.9, p=2, eps=0.05, fit_eps=False)
        m.fit(X, y)
        assert isinstance(m.eps_used_, float)
        assert m.eps_used_ == 0.05

    def test_optimal_eps_decreases_at_very_large_n(self):
        """eps should approach 0 as N -> infinity."""
        from insurance_quantile import WassersteinRobustQR

        m = WassersteinRobustQR(tau=0.9, p=2)
        eps_1e4 = m.optimal_eps(10_000)
        eps_1e6 = m.optimal_eps(1_000_000)
        assert eps_1e6 < eps_1e4

    def test_n_features_in_set_after_fit(self):
        from insurance_quantile import WassersteinRobustQR

        X, y = _make_np_data(100, 5)
        m = WassersteinRobustQR(tau=0.9, p=2, eps=0.0, fit_eps=False)
        m.fit(X, y)
        assert m.n_features_in_ == 5

    def test_tau_boundary_001_valid(self):
        """Extreme low tau should not raise during construction."""
        from insurance_quantile import WassersteinRobustQR

        m = WassersteinRobustQR(tau=0.001, p=2, eps=0.1, fit_eps=False)
        assert m.tau == 0.001

    def test_tau_boundary_999_valid(self):
        """Extreme high tau should not raise during construction."""
        from insurance_quantile import WassersteinRobustQR

        m = WassersteinRobustQR(tau=0.999, p=2, eps=0.1, fit_eps=False)
        assert m.tau == 0.999

    def test_predictions_are_finite(self):
        """All predictions should be finite for normal data."""
        from insurance_quantile import WassersteinRobustQR

        X, y = _make_np_data(200, 2)
        m = WassersteinRobustQR(tau=0.95, p=2, eps=0.1, fit_eps=False)
        m.fit(X, y)
        preds = m.predict(X)
        assert np.all(np.isfinite(preds))

    def test_intercept_is_float(self):
        from insurance_quantile import WassersteinRobustQR

        X, y = _make_np_data(100, 2)
        m = WassersteinRobustQR(tau=0.9, p=2, eps=0.05, fit_eps=False)
        m.fit(X, y)
        assert isinstance(m.intercept_, float)


# ===========================================================================
# Mathematical invariants: end-to-end
# ===========================================================================


class TestMathematicalInvariants:
    def test_quantile_predictions_monotone(self, fitted_quantile_model, exponential_data):
        """With fix_crossing=True, quantile predictions are monotone per risk."""
        X, _ = exponential_data
        X_small = X.head(100)
        preds = fitted_quantile_model.predict(X_small)
        qs = fitted_quantile_model.spec.quantiles
        cols = fitted_quantile_model.spec.column_names
        pred_matrix = np.stack([preds[c].to_numpy() for c in cols], axis=1)
        for i in range(len(qs) - 1):
            assert np.all(pred_matrix[:, i] <= pred_matrix[:, i + 1] + 1e-6), (
                f"Quantile crossing at levels {qs[i]}/{qs[i+1]}"
            )

    def test_tvar_gte_var_at_multiple_alphas(self, fitted_quantile_model, exponential_data):
        """TVaR >= VaR for alphas at 0.75 and 0.9 (coherence)."""
        from insurance_quantile import per_risk_tvar

        X, _ = exponential_data
        X_small = X.head(100)
        for alpha in [0.75, 0.9]:
            result = per_risk_tvar(fitted_quantile_model, X_small, alpha)
            tvar = result.values.to_numpy()
            var = result.var_values.to_numpy()
            assert np.all(tvar >= var - 1e-6), (
                f"TVaR < VaR at alpha={alpha}: min diff = {(tvar - var).min():.4f}"
            )

    def test_large_loss_loading_higher_at_high_alpha(
        self, fitted_quantile_model, exponential_data
    ):
        """Loading at alpha=0.99 > loading at alpha=0.75 (in expectation)."""
        from insurance_quantile import large_loss_loading

        X, _ = exponential_data
        X_small = X.head(200)

        class ConstantMean:
            def predict(self, X):
                return pl.Series("mean", np.full(len(X), 1.0))

        loading_75 = large_loss_loading(ConstantMean(), fitted_quantile_model, X_small, alpha=0.75)
        loading_99 = large_loss_loading(ConstantMean(), fitted_quantile_model, X_small, alpha=0.99)
        assert float(loading_99.mean()) > float(loading_75.mean())

    def test_pinball_loss_minimised_at_true_quantile(self):
        """
        For a large Exp(1) sample, pinball loss at the true Q(0.9) should be
        lower than at a perturbed quantile. Verifies pinball_loss is proper.
        """
        from insurance_quantile import pinball_loss

        rng = np.random.default_rng(77)
        n = 5000
        y_np = rng.exponential(1.0, n)
        y = pl.Series("y", y_np)
        true_q = -np.log(0.1)  # ≈ 2.303
        alpha = 0.9

        loss_true = pinball_loss(y, pl.Series("q", np.full(n, true_q)), alpha)
        loss_high = pinball_loss(y, pl.Series("q", np.full(n, true_q + 1.0)), alpha)
        loss_low = pinball_loss(y, pl.Series("q", np.full(n, true_q - 1.0)), alpha)

        assert loss_true < loss_high, "True quantile should have lower pinball loss than overprediction"
        assert loss_true < loss_low, "True quantile should have lower pinball loss than underprediction"

    def test_wdrqr_eps_zero_coverage_near_alpha(self):
        """
        WDRQR with eps=0 behaves like standard QR. Coverage on large test set
        should be near alpha ± 0.05.
        """
        from insurance_quantile import WassersteinRobustQR

        rng = np.random.default_rng(99)
        n_train = 1000
        n_test = 2000
        X_train = rng.uniform(0, 1, size=(n_train, 1))
        y_train = rng.exponential(size=n_train)
        X_test = rng.uniform(0, 1, size=(n_test, 1))
        y_test = rng.exponential(size=n_test)

        alpha = 0.9
        m = WassersteinRobustQR(tau=alpha, p=2, eps=0.0, fit_eps=False)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        coverage = float(np.mean(y_test <= preds))
        assert abs(coverage - alpha) < 0.05, (
            f"Coverage {coverage:.3f} too far from alpha={alpha}"
        )


# ===========================================================================
# Version export
# ===========================================================================


class TestVersionExport:
    def test_version_string_exists(self):
        import insurance_quantile

        assert hasattr(insurance_quantile, "__version__")
        assert isinstance(insurance_quantile.__version__, str)

    def test_version_not_empty(self):
        import insurance_quantile

        assert len(insurance_quantile.__version__) > 0

    def test_all_exports_accessible(self):
        """All names in __all__ should be accessible."""
        import insurance_quantile

        for name in insurance_quantile.__all__:
            # EQRN names may raise ImportError without torch, that's fine
            if name in ("EQRNModel", "EQRNDiagnostics", "GPDNet", "IntermediateQuantileEstimator"):
                continue
            assert hasattr(insurance_quantile, name), f"Missing export: {name}"
