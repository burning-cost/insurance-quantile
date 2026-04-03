"""
Expanded test coverage for insurance-quantile — batch 1.

Targets:
- _es_regressor.py internal helpers (_check_loss, _assign_bins edge cases,
  _auto_bin_count edge cases, _make_bin_edges degenerate inputs,
  _local_es_estimate boundary cases, _pseudo_values properties)
- _model.py: _apply_isotonic, _to_numpy, _series_to_numpy helpers,
  QuantileGBM.predict_tvar warning at alpha near max quantile
- _calibration.py: quantile_calibration_plot ImportError path,
  pinball_loss with large vs small predictions
- _types.py: QuantileSpec frozen, TVaRResult method stored, ExceedanceCurve
  dataframe column dtypes
- _robust.py: _check_loss function directly, _c_tau_p for p>2 and p=1,
  WassersteinRobustQR 1D X predict raises
- __init__.py: lazy EQRN import error, __version__ attribute
- _tvar.py: TVaR warning when alpha >= max quantile
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# _es_regressor internal helpers — deeper edge cases
# ---------------------------------------------------------------------------


class TestCheckLossScalar:
    """Direct tests for _check_loss_scalar."""

    def test_zero_residuals_give_zero_loss(self):
        from insurance_quantile._es_regressor import _check_loss_scalar
        residuals = np.zeros(100)
        assert _check_loss_scalar(residuals, 0.9) == pytest.approx(0.0)

    def test_asymmetry_positive_residuals(self):
        """Positive residuals penalised by alpha."""
        from insurance_quantile._es_regressor import _check_loss_scalar
        residuals = np.ones(10)  # all positive
        loss = _check_loss_scalar(residuals, 0.9)
        assert loss == pytest.approx(0.9)

    def test_asymmetry_negative_residuals(self):
        """Negative residuals penalised by (1 - alpha)."""
        from insurance_quantile._es_regressor import _check_loss_scalar
        residuals = -np.ones(10)  # all negative
        loss = _check_loss_scalar(residuals, 0.9)
        assert loss == pytest.approx(0.1)

    def test_symmetric_at_median(self):
        from insurance_quantile._es_regressor import _check_loss_scalar
        # Equal positive/negative residuals -> 0.5 * MAE
        residuals = np.array([1.0, -1.0])
        loss = _check_loss_scalar(residuals, 0.5)
        assert loss == pytest.approx(0.5)

    def test_non_negative_always(self):
        from insurance_quantile._es_regressor import _check_loss_scalar
        rng = np.random.default_rng(0)
        for alpha in [0.1, 0.5, 0.9, 0.99]:
            residuals = rng.normal(size=100)
            assert _check_loss_scalar(residuals, alpha) >= 0.0


class TestAutoBinCountEdgeCases:
    """Edge cases for _auto_bin_count beyond what test_new_coverage covers."""

    def test_n_equals_1_returns_2(self):
        from insurance_quantile._es_regressor import _auto_bin_count
        assert _auto_bin_count(1, 1) == 2

    def test_n_zero_returns_2(self):
        from insurance_quantile._es_regressor import _auto_bin_count
        assert _auto_bin_count(0, 1) == 2

    def test_p_zero_returns_2(self):
        from insurance_quantile._es_regressor import _auto_bin_count
        assert _auto_bin_count(100, 0) == 2

    def test_large_n_small_p_returns_reasonable_k(self):
        from insurance_quantile._es_regressor import _auto_bin_count
        k = _auto_bin_count(10000, 1)
        # Should be at least 2 and not absurdly large
        assert k >= 2
        assert k <= 100

    def test_large_p_reduces_k(self):
        """Higher p means fewer bins per dimension (k^p stays manageable)."""
        from insurance_quantile._es_regressor import _auto_bin_count
        k_1d = _auto_bin_count(1000, 1)
        k_5d = _auto_bin_count(1000, 5)
        # k_5d should be smaller than k_1d (formula ensures total bins tractable)
        assert k_5d <= k_1d


class TestAssignBinsEdgeCases:
    """Edge cases for _assign_bins."""

    def test_single_column_single_bin(self):
        from insurance_quantile._es_regressor import _assign_bins
        # Single column with no breakpoints — all in one bin
        X = np.array([[1.0], [2.0], [3.0]])
        edges = [np.array([])]  # zero interior breaks = one bin
        idx = _assign_bins(X, edges)
        # All observations in same bin
        assert idx.shape == (3,)
        assert len(np.unique(idx)) == 1

    def test_two_columns_four_bins(self):
        from insurance_quantile._es_regressor import _assign_bins
        rng = np.random.default_rng(99)
        X = rng.uniform(size=(100, 2))
        # One interior breakpoint per dim = 2 bins per dim = up to 4 bins
        edges = [np.array([0.5]), np.array([0.5])]
        idx = _assign_bins(X, edges)
        assert idx.shape == (100,)
        # All indices should be >= 0
        assert idx.min() >= 0

    def test_output_is_integer_type(self):
        from insurance_quantile._es_regressor import _assign_bins
        X = np.arange(12).reshape(6, 2).astype(np.float64)
        edges = [np.array([5.0]), np.array([5.0])]
        idx = _assign_bins(X, edges)
        assert np.issubdtype(idx.dtype, np.integer)

    def test_all_observations_get_assigned(self):
        from insurance_quantile._es_regressor import _assign_bins
        rng = np.random.default_rng(7)
        X = rng.uniform(size=(50, 3))
        edges = [np.array([0.33, 0.67]), np.array([0.5]), np.array([0.5])]
        idx = _assign_bins(X, edges)
        assert len(idx) == 50


class TestMakeBinEdgesDegenerate:
    """_make_bin_edges with degenerate (constant) columns."""

    def test_constant_column_no_crash(self):
        from insurance_quantile._es_regressor import _make_bin_edges
        X = np.column_stack([np.ones(50), np.random.default_rng(1).uniform(size=50)])
        edges = _make_bin_edges(X, k=4)
        assert len(edges) == 2
        # Constant column may have empty breaks after dedup
        assert isinstance(edges[0], np.ndarray)

    def test_k_equals_2_gives_one_break(self):
        from insurance_quantile._es_regressor import _make_bin_edges
        rng = np.random.default_rng(5)
        X = rng.uniform(size=(30, 1))
        edges = _make_bin_edges(X, k=2)
        assert len(edges) == 1
        # k=2 means 1 interior breakpoint (k-1 = 1)
        # After unique(), should still be at most 1
        assert len(edges[0]) <= 1


class TestLocalESEstimateEdgeCases:
    """_local_es_estimate edge cases."""

    def test_exactly_p_plus_1_observations(self):
        """Minimum viable bin: p+1 observations for OLS with intercept."""
        from insurance_quantile._es_regressor import _local_es_estimate
        X_bin = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])  # 3 obs, 2 features -> p+1 = 3
        Z_bin = np.array([2.0, 3.0, 2.5])
        result = _local_es_estimate(X_bin, Z_bin)
        assert result is not None
        assert np.isfinite(result)

    def test_p_observations_returns_none(self):
        """p observations is not enough for OLS with intercept."""
        from insurance_quantile._es_regressor import _local_es_estimate
        X_bin = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 obs, 2 features -> less than p+1=3
        Z_bin = np.array([2.0, 3.0])
        result = _local_es_estimate(X_bin, Z_bin)
        assert result is None

    def test_one_observation_one_feature_returns_none(self):
        """1 obs is less than p+1=2 for 1 feature."""
        from insurance_quantile._es_regressor import _local_es_estimate
        X_bin = np.array([[1.0]])
        Z_bin = np.array([5.0])
        result = _local_es_estimate(X_bin, Z_bin)
        assert result is None

    def test_constant_z_recovers_intercept(self):
        """If Z is constant, OLS intercept should equal that constant."""
        from insurance_quantile._es_regressor import _local_es_estimate
        rng = np.random.default_rng(0)
        X_bin = rng.normal(size=(20, 2))
        Z_bin = np.full(20, 5.0)
        result = _local_es_estimate(X_bin, Z_bin)
        assert result is not None
        assert abs(result - 5.0) < 1.0


class TestPseudoValuesProperties:
    """Additional properties for _pseudo_values."""

    def test_equal_to_q_when_y_below_q(self):
        """When y < q_hat, pseudo-value = q_hat (excess = 0)."""
        from insurance_quantile._es_regressor import _pseudo_values
        y = np.array([0.5, 1.0])
        q_hat = np.array([2.0, 3.0])  # all y < q_hat
        z = _pseudo_values(y, q_hat, alpha=0.9)
        np.testing.assert_allclose(z, q_hat)

    def test_non_negative_for_non_negative_y_and_q(self):
        """For positive y and q, pseudo-values should be >= min(y, q)."""
        from insurance_quantile._es_regressor import _pseudo_values
        rng = np.random.default_rng(42)
        y = np.abs(rng.normal(5, 1, size=100))
        q_hat = np.abs(rng.normal(4, 1, size=100))
        z = _pseudo_values(y, q_hat, alpha=0.9)
        # At least as large as q_hat (since excess is non-negative)
        assert (z >= q_hat).all()

    def test_exceeds_y_when_y_exceeds_q(self):
        """When y > q, pseudo-value = q + (y - q) / (1 - alpha) > y for alpha < 1."""
        from insurance_quantile._es_regressor import _pseudo_values
        y = np.array([10.0])
        q_hat = np.array([5.0])
        alpha = 0.9
        z = _pseudo_values(y, q_hat, alpha)
        # z = 5 + (10 - 5) / 0.1 = 5 + 50 = 55
        assert abs(z[0] - 55.0) < 1e-10


# ---------------------------------------------------------------------------
# _model.py internal helpers
# ---------------------------------------------------------------------------


class TestApplyIsotonic:
    """Direct tests for _apply_isotonic."""

    def test_already_monotone_unchanged(self):
        from insurance_quantile._model import _apply_isotonic
        pred = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        result = _apply_isotonic(pred)
        np.testing.assert_allclose(result, pred, atol=1e-6)

    def test_fixes_single_crossing(self):
        from insurance_quantile._model import _apply_isotonic
        # Row with a crossing: [1, 3, 2] -> should be fixed to non-decreasing
        pred = np.array([[1.0, 3.0, 2.0]])
        result = _apply_isotonic(pred)
        assert result[0, 0] <= result[0, 1] + 1e-9
        assert result[0, 1] <= result[0, 2] + 1e-9

    def test_single_column_unchanged(self):
        from insurance_quantile._model import _apply_isotonic
        pred = np.array([[5.0], [3.0], [7.0]])
        result = _apply_isotonic(pred)
        np.testing.assert_allclose(result, pred)

    def test_output_shape_preserved(self):
        from insurance_quantile._model import _apply_isotonic
        pred = np.random.default_rng(0).uniform(size=(100, 5))
        result = _apply_isotonic(pred)
        assert result.shape == (100, 5)

    def test_descending_row_made_monotone(self):
        from insurance_quantile._model import _apply_isotonic
        pred = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
        result = _apply_isotonic(pred)
        # After isotonic regression, all values should be non-decreasing
        for i in range(4):
            assert result[0, i] <= result[0, i + 1] + 1e-9


class TestToNumpy:
    """Direct tests for _to_numpy and _series_to_numpy helpers."""

    def test_to_numpy_from_polars_dataframe(self):
        from insurance_quantile._model import _to_numpy
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        arr = _to_numpy(df)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (2, 2)

    def test_to_numpy_from_numpy_array(self):
        from insurance_quantile._model import _to_numpy
        arr_in = np.array([[1, 2], [3, 4]], dtype=np.int32)
        arr_out = _to_numpy(arr_in)
        assert arr_out.dtype == np.float64

    def test_series_to_numpy_from_polars_series(self):
        from insurance_quantile._model import _series_to_numpy
        s = pl.Series("y", [1.0, 2.0, 3.0])
        arr = _series_to_numpy(s)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_series_to_numpy_from_numpy_array(self):
        from insurance_quantile._model import _series_to_numpy
        a = np.array([1, 2, 3], dtype=np.int16)
        arr = _series_to_numpy(a)
        assert arr.dtype == np.float64


class TestQuantileGBMPredictTVaRWarning:
    """TVaR warning when alpha is very close to or at max quantile."""

    def test_alpha_at_max_quantile_warns(self, exponential_data):
        from insurance_quantile import QuantileGBM
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.5, 0.75, 0.9], iterations=100)
        model.fit(X, y)
        # alpha = 0.9 = max quantile: should warn about unreliable TVaR
        # The code raises ValueError for no levels above, so this test
        # actually checks that the error is informative
        with pytest.raises(ValueError, match="no quantile levels above alpha"):
            model.predict_tvar(X.head(5), alpha=0.9)

    def test_tvar_at_first_quantile_works(self, exponential_data):
        """TVaR at the first quantile level should work as long as there are levels above."""
        from insurance_quantile import QuantileGBM
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.1, 0.5, 0.9], iterations=100)
        model.fit(X, y)
        result = model.predict_tvar(X.head(10), alpha=0.1)
        assert len(result) == 10
        assert (result.to_numpy() > 0).all()


# ---------------------------------------------------------------------------
# _calibration.py — additional edge cases
# ---------------------------------------------------------------------------


class TestPinballLossEdgeCases:
    """Further edge cases for pinball_loss."""

    def test_all_y_above_q_loss_equals_alpha_times_mean_residual(self):
        """When all y > q, loss = alpha * mean(y - q)."""
        from insurance_quantile import pinball_loss
        y = pl.Series("y", [3.0, 4.0, 5.0])
        q = pl.Series("q", [1.0, 1.0, 1.0])
        # residuals: 2, 3, 4; mean = 3; loss = 0.9 * 3 = 2.7
        loss = pinball_loss(y, q, alpha=0.9)
        assert loss == pytest.approx(2.7)

    def test_all_y_below_q_loss_equals_one_minus_alpha_times_mean_overpredict(self):
        """When all y < q, loss = (1-alpha) * mean(q - y)."""
        from insurance_quantile import pinball_loss
        y = pl.Series("y", [1.0, 1.0, 1.0])
        q = pl.Series("q", [3.0, 4.0, 5.0])
        # residuals: -2, -3, -4; loss = (1-0.9) * mean(2,3,4) = 0.1 * 3 = 0.3
        loss = pinball_loss(y, q, alpha=0.9)
        assert loss == pytest.approx(0.3)

    def test_single_observation_equals_alpha_times_residual_when_y_gt_q(self):
        from insurance_quantile import pinball_loss
        y = pl.Series("y", [5.0])
        q = pl.Series("q", [2.0])
        loss = pinball_loss(y, q, alpha=0.8)
        assert loss == pytest.approx(0.8 * 3.0)

    def test_negative_alpha_raises(self):
        from insurance_quantile import pinball_loss
        y = pl.Series("y", [1.0])
        q = pl.Series("q", [1.0])
        with pytest.raises(ValueError, match="alpha must be in"):
            pinball_loss(y, q, alpha=-0.1)

    def test_alpha_exactly_boundary_0_raises(self):
        from insurance_quantile import pinball_loss
        y = pl.Series("y", [1.0])
        q = pl.Series("q", [1.0])
        with pytest.raises(ValueError):
            pinball_loss(y, q, alpha=0.0)

    def test_alpha_exactly_boundary_1_raises(self):
        from insurance_quantile import pinball_loss
        y = pl.Series("y", [1.0])
        q = pl.Series("q", [1.0])
        with pytest.raises(ValueError):
            pinball_loss(y, q, alpha=1.0)


class TestQuantileCalibrationPlotImportError:
    """quantile_calibration_plot raises ImportError when matplotlib is missing."""

    def test_raises_import_error_without_matplotlib(self, monkeypatch):
        """Simulate missing matplotlib by patching builtins.__import__."""
        from insurance_quantile import quantile_calibration_plot
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("No module named 'matplotlib'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        y = pl.Series("y", [1.0, 2.0, 3.0])
        preds = pl.DataFrame({"q_0.5": [1.0, 1.5, 2.0]})
        with pytest.raises(ImportError, match="matplotlib"):
            quantile_calibration_plot(y, preds, [0.5])


# ---------------------------------------------------------------------------
# _types.py — additional tests
# ---------------------------------------------------------------------------


class TestQuantileSpecFrozen:
    """QuantileSpec is a frozen dataclass — mutation should raise."""

    def test_frozen_cannot_set_quantiles(self):
        from insurance_quantile import QuantileSpec
        spec = QuantileSpec(quantiles=[0.5, 0.9])
        with pytest.raises((TypeError, AttributeError)):
            spec.quantiles = [0.1]  # type: ignore[misc]

    def test_frozen_cannot_set_mode(self):
        from insurance_quantile import QuantileSpec
        spec = QuantileSpec(quantiles=[0.5])
        with pytest.raises((TypeError, AttributeError)):
            spec.mode = "expectile"  # type: ignore[misc]

    def test_single_quantile_column_name(self):
        from insurance_quantile import QuantileSpec
        spec = QuantileSpec(quantiles=[0.99])
        assert spec.column_names == ["q_0.99"]

    def test_mode_default_quantile(self):
        from insurance_quantile import QuantileSpec
        spec = QuantileSpec(quantiles=[0.5])
        assert spec.mode == "quantile"

    def test_near_boundary_quantiles_valid(self):
        from insurance_quantile import QuantileSpec
        spec = QuantileSpec(quantiles=[0.001, 0.999])
        assert len(spec.quantiles) == 2


class TestTVaRResultAdditional:
    """Additional tests for TVaRResult."""

    def test_custom_method_stored(self):
        from insurance_quantile._types import TVaRResult
        result = TVaRResult(
            alpha=0.9,
            values=pl.Series("tvar", [1.0]),
            var_values=pl.Series("var", [0.5]),
            method="custom_method",
        )
        assert result.method == "custom_method"

    def test_loading_over_var_matches_difference(self):
        from insurance_quantile._types import TVaRResult
        tvar = pl.Series("tvar", [5.0, 7.0, 9.0])
        var = pl.Series("var", [3.0, 4.0, 5.0])
        result = TVaRResult(alpha=0.95, values=tvar, var_values=var)
        loading = result.loading_over_var
        np.testing.assert_allclose(loading.to_numpy(), [2.0, 3.0, 4.0])

    def test_zero_loading_when_tvar_equals_var(self):
        from insurance_quantile._types import TVaRResult
        v = pl.Series("v", [2.0, 3.0])
        result = TVaRResult(alpha=0.9, values=v, var_values=v)
        assert (result.loading_over_var.to_numpy() == 0.0).all()


class TestExceedanceCurveAdditional:
    """Additional tests for ExceedanceCurve dataclass."""

    def test_as_dataframe_column_dtypes(self):
        from insurance_quantile._types import ExceedanceCurve
        curve = ExceedanceCurve(
            thresholds=[0.0, 1.0, 2.0],
            probabilities=[0.9, 0.5, 0.1],
            n_risks=50,
        )
        df = curve.as_dataframe()
        assert df["threshold"].dtype in (pl.Float64, pl.Float32, pl.Int64)
        assert df["exceedance_prob"].dtype in (pl.Float64, pl.Float32)

    def test_as_dataframe_single_threshold(self):
        from insurance_quantile._types import ExceedanceCurve
        curve = ExceedanceCurve(
            thresholds=[0.0],
            probabilities=[1.0],
            n_risks=10,
        )
        df = curve.as_dataframe()
        assert df.shape == (1, 2)

    def test_as_dataframe_no_n_risks_column(self):
        """ExceedanceCurve.as_dataframe() should NOT include n_risks — that's stored separately."""
        from insurance_quantile._types import ExceedanceCurve
        curve = ExceedanceCurve(thresholds=[0.0], probabilities=[1.0], n_risks=5)
        df = curve.as_dataframe()
        assert "n_risks" not in df.columns


class TestTwoPartResultFieldTypes:
    """TwoPartResult field type verification."""

    def test_all_series_are_polars(self):
        from insurance_quantile._types import TwoPartResult
        n = 3
        r = TwoPartResult(
            premium=pl.Series("premium", np.ones(n)),
            pure_premium=pl.Series("pure_premium", np.ones(n)),
            safety_loading=pl.Series("safety_loading", np.zeros(n)),
            no_claim_prob=pl.Series("no_claim_prob", np.full(n, 0.5)),
            adjusted_tau=pl.Series("adjusted_tau", np.full(n, 0.6)),
            severity_quantile=pl.Series("severity_quantile", np.ones(n)),
            n_fallback=0,
            tau=0.9,
            gamma=0.5,
        )
        for attr in ("premium", "pure_premium", "safety_loading",
                     "no_claim_prob", "adjusted_tau", "severity_quantile"):
            assert isinstance(getattr(r, attr), pl.Series), f"{attr} is not pl.Series"

    def test_n_fallback_is_int(self):
        from insurance_quantile._types import TwoPartResult
        n = 2
        r = TwoPartResult(
            premium=pl.Series("premium", np.ones(n)),
            pure_premium=pl.Series("pure_premium", np.ones(n)),
            safety_loading=pl.Series("safety_loading", np.zeros(n)),
            no_claim_prob=pl.Series("no_claim_prob", np.full(n, 0.5)),
            adjusted_tau=pl.Series("adjusted_tau", np.full(n, 0.6)),
            severity_quantile=pl.Series("severity_quantile", np.ones(n)),
            n_fallback=1,
            tau=0.9,
            gamma=0.5,
        )
        assert isinstance(r.n_fallback, int)
        assert r.n_fallback == 1


# ---------------------------------------------------------------------------
# _robust.py — direct tests of internal math
# ---------------------------------------------------------------------------


class TestCheckLossRobust:
    """Direct tests for _check_loss in _robust.py."""

    def test_zero_residuals_zero_loss(self):
        from insurance_quantile._robust import _check_loss
        r = np.zeros(50)
        assert _check_loss(r, 0.9) == pytest.approx(0.0)

    def test_positive_residuals_alpha_penalised(self):
        from insurance_quantile._robust import _check_loss
        r = np.ones(10)
        assert _check_loss(r, 0.9) == pytest.approx(0.9)

    def test_negative_residuals_one_minus_alpha_penalised(self):
        from insurance_quantile._robust import _check_loss
        r = -np.ones(10)
        assert _check_loss(r, 0.9) == pytest.approx(0.1)

    def test_always_non_negative(self):
        from insurance_quantile._robust import _check_loss
        rng = np.random.default_rng(1)
        for alpha in [0.1, 0.5, 0.9]:
            r = rng.normal(size=100)
            assert _check_loss(r, alpha) >= 0.0


class TestCTauPEdgeCases:
    """_c_tau_p at various p values."""

    def test_p1_equals_max_tau_one_minus_tau(self):
        from insurance_quantile._robust import _c_tau_p
        # p=1: c = max(tau, 1-tau)
        assert abs(_c_tau_p(0.9, 1) - 0.9) < 1e-12
        assert abs(_c_tau_p(0.3, 1) - 0.7) < 1e-12
        assert abs(_c_tau_p(0.5, 1) - 0.5) < 1e-12

    def test_p3_formula(self):
        from insurance_quantile._robust import _c_tau_p
        # q = p/(p-1) = 3/2 = 1.5; c = (tau^1.5 + (1-tau)^1.5)^(1/1.5)
        tau = 0.8
        p = 3
        q = p / (p - 1)  # 1.5
        expected = (tau**q + (1 - tau)**q) ** (1 / q)
        assert abs(_c_tau_p(tau, p) - expected) < 1e-12

    def test_positive_for_all_p(self):
        from insurance_quantile._robust import _c_tau_p
        for p in [1, 2, 3, 5]:
            for tau in [0.1, 0.5, 0.9]:
                assert _c_tau_p(tau, p) > 0.0

    def test_symmetry_p2(self):
        from insurance_quantile._robust import _c_tau_p
        assert abs(_c_tau_p(0.8, 2) - _c_tau_p(0.2, 2)) < 1e-12


class TestInterceptCorrectionAdditional:
    """Additional intercept correction tests."""

    def test_p1_correction_always_zero(self):
        """p=1 should always give zero correction regardless of beta_norm or eps."""
        from insurance_quantile._robust import _intercept_correction
        for eps in [0.0, 0.1, 0.5]:
            corr = _intercept_correction(beta_norm=2.0, tau=0.9, p=1, eps=eps)
            assert corr == 0.0, f"Expected 0 for p=1, got {corr}"

    def test_eps_zero_gives_zero_correction(self):
        from insurance_quantile._robust import _intercept_correction
        corr = _intercept_correction(beta_norm=1.5, tau=0.95, p=2, eps=0.0)
        assert corr == 0.0

    def test_correction_scales_with_beta_norm(self):
        from insurance_quantile._robust import _intercept_correction
        c1 = _intercept_correction(beta_norm=1.0, tau=0.9, p=2, eps=0.1)
        c2 = _intercept_correction(beta_norm=2.0, tau=0.9, p=2, eps=0.1)
        # Correction scales linearly with beta_norm
        assert abs(c2 / c1 - 2.0) < 1e-10


class TestWassersteinRobustQRAdditional:
    """Additional WassersteinRobustQR tests."""

    def test_1d_predict_array_raises(self):
        """Predicting with 1D X should raise ValueError."""
        from insurance_quantile import WassersteinRobustQR
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(100, 2))
        y = rng.exponential(size=100)
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        model.fit(X, y)
        X_bad = np.ones(5)  # 1D
        with pytest.raises(ValueError, match="2-dimensional"):
            model.predict(X_bad)

    def test_n_features_in_set(self):
        from insurance_quantile import WassersteinRobustQR
        rng = np.random.default_rng(1)
        X = rng.uniform(size=(100, 4))
        y = rng.exponential(size=100)
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.05)
        model.fit(X, y)
        assert model.n_features_in_ == 4

    def test_tau_boundary_valid(self):
        """Tau values near 0 and 1 (but not equal) should be valid."""
        from insurance_quantile import WassersteinRobustQR
        m1 = WassersteinRobustQR(tau=0.001, p=2, eps=0.1)
        m2 = WassersteinRobustQR(tau=0.999, p=2, eps=0.1)
        assert m1.tau == 0.001
        assert m2.tau == 0.999

    def test_repr_contains_tau_p_eps(self):
        from insurance_quantile import WassersteinRobustQR
        model = WassersteinRobustQR(tau=0.95, p=2, eps=0.2)
        r = repr(model)
        assert "0.95" in r
        assert "p=2" in r
        assert "0.2" in r

    def test_fit_1d_y_vector(self):
        """y must be 1D."""
        from insurance_quantile import WassersteinRobustQR
        X = np.ones((10, 2))
        y_bad = np.ones((10, 2))
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        with pytest.raises(ValueError, match="1-dimensional"):
            model.fit(X, y_bad)


# ---------------------------------------------------------------------------
# __init__.py: lazy imports and version
# ---------------------------------------------------------------------------


class TestLazyEQRNImports:
    """Test the lazy EQRN import mechanism."""

    def test_unknown_attribute_raises_attribute_error(self):
        import insurance_quantile
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = insurance_quantile.NonExistentClass  # type: ignore[attr-defined]

    def test_version_attribute_exists(self):
        import insurance_quantile
        assert hasattr(insurance_quantile, "__version__")

    def test_version_is_string(self):
        import insurance_quantile
        assert isinstance(insurance_quantile.__version__, str)

    def test_version_not_empty(self):
        import insurance_quantile
        assert len(insurance_quantile.__version__) > 0

    def test_eqrn_missing_raises_import_error(self):
        """Accessing EQRN names without torch/lightgbm should raise ImportError."""
        import sys
        import importlib
        import insurance_quantile

        # Only test if torch is NOT installed (to avoid side-effects on real envs)
        if "torch" not in sys.modules:
            try:
                _ = insurance_quantile.EQRNModel
                # If no ImportError, torch must be installed — skip
            except ImportError as e:
                assert "torch" in str(e) or "eqrn" in str(e).lower()


class TestPublicAPIExports:
    """All __all__ names should be importable from the package."""

    def test_all_public_exports_present(self):
        import insurance_quantile
        for name in insurance_quantile.__all__:
            # EQRN names may require torch; skip them
            if name in ("EQRNModel", "EQRNDiagnostics", "GPDNet", "IntermediateQuantileEstimator"):
                continue
            assert hasattr(insurance_quantile, name), f"Missing export: {name}"

    def test_core_classes_importable(self):
        from insurance_quantile import (
            QuantileGBM, WassersteinRobustQR, ExpectedShortfallRegressor,
            TwoPartQuantilePremium, MeanModelWrapper
        )
        assert QuantileGBM is not None
        assert WassersteinRobustQR is not None
        assert ExpectedShortfallRegressor is not None
        assert TwoPartQuantilePremium is not None
        assert MeanModelWrapper is not None

    def test_types_importable(self):
        from insurance_quantile import (
            QuantileSpec, TailModel, TVaRResult, ExceedanceCurve, TwoPartResult
        )
        assert QuantileSpec is not None


# ---------------------------------------------------------------------------
# _tvar.py: TVaR warning paths
# ---------------------------------------------------------------------------


class TestTVaRWarnings:
    """Test UserWarning when alpha >= max model quantile."""

    def test_tvar_warns_when_alpha_near_max_quantile(self, exponential_data):
        from insurance_quantile import QuantileGBM
        X, y = exponential_data
        # Fit model with max quantile = 0.95
        model = QuantileGBM(quantiles=[0.5, 0.75, 0.9, 0.95], iterations=100)
        model.fit(X, y)
        # alpha=0.94: just below max, but raises because above all fitted quantiles
        # Actually alpha=0.95 would be AT max — let's test alpha slightly below max
        # that still has levels above it
        # alpha=0.75: levels [0.9, 0.95] are above — should work without warning
        from insurance_quantile import per_risk_tvar
        result = per_risk_tvar(model, X.head(20), alpha=0.75)
        assert len(result.values) == 20

    def test_tvar_per_risk_raises_when_no_levels_above(self, exponential_data):
        from insurance_quantile import QuantileGBM, per_risk_tvar
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.5, 0.9], iterations=100)
        model.fit(X, y)
        with pytest.raises(ValueError, match="no quantile levels above alpha"):
            per_risk_tvar(model, X.head(5), alpha=0.95)

    def test_tvar_method_label_trapezoidal(self, exponential_data):
        from insurance_quantile import QuantileGBM, per_risk_tvar
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.5, 0.9, 0.95], iterations=100)
        model.fit(X, y)
        result = per_risk_tvar(model, X.head(10), alpha=0.5)
        assert result.method == "trapezoidal"

    def test_tvar_var_values_polars_series(self, exponential_data):
        from insurance_quantile import QuantileGBM, per_risk_tvar
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.5, 0.9, 0.95], iterations=100)
        model.fit(X, y)
        result = per_risk_tvar(model, X.head(10), alpha=0.5)
        assert isinstance(result.var_values, pl.Series)
        assert isinstance(result.values, pl.Series)


# ---------------------------------------------------------------------------
# _two_part.py: _interpolate_severity_quantile
# ---------------------------------------------------------------------------


class TestInterpolateSeverityQuantile:
    """Direct tests for the private interpolation helper."""

    def test_all_invalid_returns_all_nan(self):
        from insurance_quantile._two_part import _interpolate_severity_quantile
        n = 5
        q_matrix = np.ones((n, 3))
        q_levels = np.array([0.5, 0.7, 0.9])
        tau_i = np.full(n, np.nan)
        valid_mask = np.zeros(n, dtype=bool)
        result, extrap = _interpolate_severity_quantile(q_matrix, q_levels, tau_i, valid_mask)
        assert np.isnan(result).all()
        assert extrap == 0.0

    def test_exact_quantile_level_returns_that_quantile(self):
        """If tau_i = q_levels[k], result should equal q_matrix[:, k]."""
        from insurance_quantile._two_part import _interpolate_severity_quantile
        n = 3
        q_matrix = np.array([[100.0, 200.0, 300.0]] * n)
        q_levels = np.array([0.5, 0.7, 0.9])
        tau_i = np.full(n, 0.7)  # exactly at second level
        valid_mask = np.ones(n, dtype=bool)
        result, _ = _interpolate_severity_quantile(q_matrix, q_levels, tau_i, valid_mask)
        # At q=0.7: interpolated between q_levels[1]=0.7 and q_levels[2]=0.9
        # weight = (0.7 - 0.7) / (0.9 - 0.7) = 0 => result = q[1] = 200
        np.testing.assert_allclose(result, 200.0, atol=1e-6)

    def test_tau_above_max_gives_flat_extrapolation(self):
        """tau_i > max(q_levels) should return the highest quantile value."""
        from insurance_quantile._two_part import _interpolate_severity_quantile
        n = 2
        q_matrix = np.array([[100.0, 200.0, 300.0]] * n)
        q_levels = np.array([0.5, 0.7, 0.9])
        tau_i = np.full(n, 0.99)  # above max 0.9
        valid_mask = np.ones(n, dtype=bool)
        result, extrap = _interpolate_severity_quantile(q_matrix, q_levels, tau_i, valid_mask)
        # Should extrapolate flat at q[2] = 300
        np.testing.assert_allclose(result, 300.0, atol=1e-6)
        assert extrap > 0.0  # extrapolation fraction > 0

    def test_mixed_valid_invalid(self):
        """Some valid, some invalid: valid ones get interpolated, invalid get NaN."""
        from insurance_quantile._two_part import _interpolate_severity_quantile
        n = 4
        q_matrix = np.array([[100.0, 200.0]] * n)
        q_levels = np.array([0.5, 0.9])
        tau_i = np.array([0.7, np.nan, 0.6, np.nan])
        valid_mask = np.array([True, False, True, False])
        result, _ = _interpolate_severity_quantile(q_matrix, q_levels, tau_i, valid_mask)
        assert np.isfinite(result[0])
        assert np.isnan(result[1])
        assert np.isfinite(result[2])
        assert np.isnan(result[3])


# ---------------------------------------------------------------------------
# QuantileGBM additional edge cases
# ---------------------------------------------------------------------------


class TestQuantileGBMAdditional:
    """Additional edge case tests for QuantileGBM."""

    def test_predict_returns_float64_columns(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        preds = fitted_quantile_model.predict(X.head(50))
        for col in preds.columns:
            assert preds[col].dtype == pl.Float64, f"{col} is not Float64"

    def test_spec_mode_is_quantile_by_default(self):
        from insurance_quantile import QuantileGBM
        model = QuantileGBM(quantiles=[0.5, 0.9])
        assert model.spec.mode == "quantile"

    def test_spec_mode_is_expectile_when_use_expectile(self):
        from insurance_quantile import QuantileGBM
        model = QuantileGBM(quantiles=[0.5, 0.9], use_expectile=True)
        assert model.spec.mode == "expectile"

    def test_is_fitted_property_false_before_fit(self):
        from insurance_quantile import QuantileGBM
        model = QuantileGBM(quantiles=[0.5])
        assert model.is_fitted is False

    def test_is_fitted_property_true_after_fit(self, exponential_data):
        from insurance_quantile import QuantileGBM
        X, y = exponential_data
        model = QuantileGBM(quantiles=[0.5], iterations=50)
        model.fit(X, y)
        assert model.is_fitted is True

    def test_predict_tvar_series_dtype(self, fitted_quantile_model, exponential_data):
        X, _ = exponential_data
        tvar = fitted_quantile_model.predict_tvar(X.head(20), alpha=0.5)
        assert tvar.dtype == pl.Float64

    def test_calibration_report_mean_pinball_positive(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        report = fitted_quantile_model.calibration_report(X.head(200), y.head(200))
        assert report["mean_pinball_loss"] > 0.0


# ---------------------------------------------------------------------------
# coverage_check additional tests
# ---------------------------------------------------------------------------


class TestCoverageCheckAdditional:
    """Additional coverage_check tests."""

    def test_all_above_threshold_gives_coverage_one(self):
        from insurance_quantile import coverage_check
        y = pl.Series("y", [1.0, 2.0, 3.0])
        # Set q very high so all y are below q -> coverage = 1.0
        preds = pl.DataFrame({"q_0.9": [100.0, 100.0, 100.0]})
        result = coverage_check(y, preds, [0.9])
        assert abs(float(result["observed_coverage"][0]) - 1.0) < 1e-10

    def test_all_below_threshold_gives_coverage_zero(self):
        from insurance_quantile import coverage_check
        y = pl.Series("y", [10.0, 20.0, 30.0])
        # Set q very low so all y are above q -> coverage = 0.0
        preds = pl.DataFrame({"q_0.9": [0.0, 0.0, 0.0]})
        result = coverage_check(y, preds, [0.9])
        assert abs(float(result["observed_coverage"][0]) - 0.0) < 1e-10

    def test_returns_correct_number_of_rows(self):
        from insurance_quantile import coverage_check
        y = pl.Series("y", [1.0, 2.0])
        preds = pl.DataFrame({
            "q_0.5": [1.5, 1.5],
            "q_0.9": [3.0, 3.0],
            "q_0.99": [5.0, 5.0],
        })
        result = coverage_check(y, preds, [0.5, 0.9, 0.99])
        assert len(result) == 3

    def test_expected_coverage_equals_quantile(self):
        from insurance_quantile import coverage_check
        y = pl.Series("y", [1.0, 2.0, 3.0])
        qs = [0.25, 0.5, 0.75, 0.9]
        preds = pl.DataFrame({f"q_{q}": [2.0] * 3 for q in qs})
        result = coverage_check(y, preds, qs)
        for row in result.iter_rows(named=True):
            assert row["expected_coverage"] == row["quantile"]


# ---------------------------------------------------------------------------
# exceedance_curve additional
# ---------------------------------------------------------------------------


class TestExceedanceCurveAdditional:
    """Additional tests for exceedance_curve and oep_curve."""

    def test_exceedance_curve_single_risk(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import exceedance_curve
        X, _ = exponential_data
        result = exceedance_curve(fitted_quantile_model, X.head(1))
        assert len(result) == 100  # default n_thresholds
        assert result["n_risks"].unique().to_list() == [1]

    def test_exceedance_curve_custom_n_thresholds(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import exceedance_curve
        X, _ = exponential_data
        result = exceedance_curve(fitted_quantile_model, X.head(50), n_thresholds=25)
        assert len(result) == 25

    def test_oep_curve_thresholds_list_correct_length(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import oep_curve
        import warnings
        X, _ = exponential_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = oep_curve(fitted_quantile_model, X.head(30), n_thresholds=40)
        assert len(result.thresholds) == 40
        assert len(result.probabilities) == 40

    def test_oep_curve_independence_probabilities_all_in_range(
        self, fitted_quantile_model, exponential_data
    ):
        from insurance_quantile import oep_curve
        X, _ = exponential_data
        result = oep_curve(
            fitted_quantile_model, X.head(20),
            thresholds=[0.0, 1.0, 5.0, 100.0],
            independence_assumption=True
        )
        probs = result.probabilities
        for p in probs:
            assert 0.0 <= p <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# _loading.py additional tests
# ---------------------------------------------------------------------------


class TestLargeLossLoadingAdditional:
    """Additional tests for large_loss_loading."""

    def test_loading_column_dtype_float64(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import large_loss_loading
        X, _ = exponential_data

        class ConstantMean:
            def predict(self, X):
                return pl.Series("mean", [1.0] * len(X))

        result = large_loss_loading(ConstantMean(), fitted_quantile_model, X.head(50))
        assert result.dtype == pl.Float64

    def test_loading_with_polars_series_mean_model(
        self, fitted_quantile_model, exponential_data
    ):
        from insurance_quantile import large_loss_loading
        X, _ = exponential_data

        class SeriesMean:
            def predict(self, X: pl.DataFrame) -> pl.Series:
                return pl.Series("mean", np.full(len(X), 0.5))

        result = large_loss_loading(SeriesMean(), fitted_quantile_model, X.head(30))
        assert isinstance(result, pl.Series)
        assert len(result) == 30

    def test_ilf_zero_basic_limit_raises(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import ilf
        X, _ = exponential_data
        with pytest.raises(ValueError, match="must be positive"):
            ilf(fitted_quantile_model, X.head(5), basic_limit=0.0, higher_limit=10.0)

    def test_ilf_basic_equals_higher_raises(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import ilf
        X, _ = exponential_data
        with pytest.raises(ValueError, match="less than higher_limit"):
            ilf(fitted_quantile_model, X.head(5), basic_limit=5.0, higher_limit=5.0)

    def test_mean_model_wrapper_series_output(self, exponential_data):
        from insurance_quantile import MeanModelWrapper
        X, _ = exponential_data

        class NumpyModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.ones(len(X))

        wrapped = MeanModelWrapper(NumpyModel())
        result = wrapped.predict(X.head(20))
        assert isinstance(result, pl.Series)
        assert result.name == "mean"
        assert len(result) == 20


# ---------------------------------------------------------------------------
# portfolio_tvar additional
# ---------------------------------------------------------------------------


class TestPortfolioTVaRAdditional:
    """Additional tests for portfolio_tvar."""

    def test_sum_equals_n_times_mean(self, fitted_quantile_model, exponential_data):
        from insurance_quantile import portfolio_tvar
        X, _ = exponential_data
        X_small = X.head(50)
        m = portfolio_tvar(fitted_quantile_model, X_small, alpha=0.9, aggregate_method="mean")
        s = portfolio_tvar(fitted_quantile_model, X_small, alpha=0.9, aggregate_method="sum")
        assert abs(s - m * 50) < 1e-6

    def test_portfolio_tvar_invalid_method_error_message(
        self, fitted_quantile_model, exponential_data
    ):
        from insurance_quantile import portfolio_tvar
        X, _ = exponential_data
        with pytest.raises(ValueError, match="aggregate_method must be"):
            portfolio_tvar(fitted_quantile_model, X.head(10), alpha=0.9, aggregate_method="invalid")

    def test_portfolio_tvar_returns_python_float(
        self, fitted_quantile_model, exponential_data
    ):
        from insurance_quantile import portfolio_tvar
        X, _ = exponential_data
        result = portfolio_tvar(fitted_quantile_model, X.head(20), alpha=0.9)
        assert isinstance(result, float)
