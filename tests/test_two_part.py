"""
Tests for TwoPartQuantilePremium and TwoPartResult.

Uses mock objects to avoid model fitting in unit tests — tests are
fast, deterministic, and verify the mathematics rather than model quality.

The integration test (TestTwoPartIntegration) does fit real models on
synthetic data and is the only test that touches CatBoost / sklearn fit.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_quantile import TwoPartQuantilePremium, TwoPartResult, QuantileGBM


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockFreqModel:
    """
    Mock binary classifier that returns fixed no-claim probabilities.

    classes_[0] = 0 (no-claim), classes_[1] = 1 (claim).
    predict_proba returns shape (n, 2): [p_no_claim, p_claim].
    """

    classes_ = np.array([0, 1])

    def __init__(self, p_no_claim: np.ndarray):
        self._p = np.asarray(p_no_claim, dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        p = self._p if len(self._p) == n else np.full(n, self._p[0])
        return np.column_stack([p, 1.0 - p])


class _MockMeanSevModel:
    """Mock mean severity model returning fixed values."""

    def __init__(self, mean_sev: float | np.ndarray):
        self._mean = mean_sev

    def predict(self, X) -> np.ndarray:
        n = len(X)
        if np.isscalar(self._mean):
            return np.full(n, float(self._mean))
        return np.asarray(self._mean, dtype=np.float64)


def _make_mock_sev_model(
    quantiles: list[float],
    q_values: np.ndarray,
    n_policies: int,
) -> QuantileGBM:
    """
    Build a QuantileGBM-like object whose predict() returns fixed quantile values.

    Avoids fitting a real CatBoost model by patching predict() after creating
    a minimal fitted state.
    """
    model = QuantileGBM(quantiles=quantiles, iterations=1)

    # Patch the model to return fixed predictions without fitting
    q_df = pl.DataFrame(
        {f"q_{q}": np.full(n_policies, q_values[i]) for i, q in enumerate(quantiles)}
    )

    class _PatchedPredict:
        def __call__(self, X):
            return q_df

    model._is_fitted = True
    model.predict = _PatchedPredict()

    # Also need metadata for is_fitted check
    from insurance_quantile._types import TailModel, QuantileSpec
    model._metadata = TailModel(
        spec=model._spec,
        n_features=1,
        feature_names=["x"],
        n_training_rows=n_policies,
    )
    return model


def _make_test_X(n: int = 2) -> pl.DataFrame:
    """Minimal feature matrix for mock tests."""
    return pl.DataFrame({"x": np.ones(n)})


# ---------------------------------------------------------------------------
# TestTwoPartResult: dataclass smoke tests
# ---------------------------------------------------------------------------

class TestTwoPartResult:
    def test_instantiation(self):
        n = 5
        result = TwoPartResult(
            premium=pl.Series("premium", np.ones(n)),
            pure_premium=pl.Series("pure_premium", np.ones(n)),
            safety_loading=pl.Series("safety_loading", np.zeros(n)),
            no_claim_prob=pl.Series("no_claim_prob", np.full(n, 0.8)),
            adjusted_tau=pl.Series("adjusted_tau", np.full(n, 0.5)),
            severity_quantile=pl.Series("severity_quantile", np.ones(n)),
            n_fallback=0,
            tau=0.9,
            gamma=0.5,
        )
        assert result.n_fallback == 0
        assert result.tau == 0.9
        assert result.gamma == 0.5
        assert len(result.premium) == n

    def test_series_names(self):
        n = 3
        result = TwoPartResult(
            premium=pl.Series("premium", np.ones(n)),
            pure_premium=pl.Series("pure_premium", np.ones(n)),
            safety_loading=pl.Series("safety_loading", np.zeros(n)),
            no_claim_prob=pl.Series("no_claim_prob", np.full(n, 0.8)),
            adjusted_tau=pl.Series("adjusted_tau", np.full(n, 0.5)),
            severity_quantile=pl.Series("severity_quantile", np.ones(n)),
            n_fallback=0,
            tau=0.9,
            gamma=0.5,
        )
        assert result.premium.name == "premium"
        assert result.safety_loading.name == "safety_loading"


# ---------------------------------------------------------------------------
# TestInputValidation: ValueError / RuntimeError on bad inputs
# ---------------------------------------------------------------------------

class TestInputValidation:
    def _build_tpqp(self, n=2):
        sev = _make_mock_sev_model([0.6, 0.7], np.array([800.0, 1000.0]), n)
        freq = _MockFreqModel(np.array([0.7, 0.7]))
        return TwoPartQuantilePremium(freq, sev)

    def test_tau_zero_raises(self):
        tpqp = self._build_tpqp()
        with pytest.raises(ValueError, match="tau must be strictly in"):
            tpqp.predict_premium(_make_test_X(2), tau=0.0)

    def test_tau_one_raises(self):
        tpqp = self._build_tpqp()
        with pytest.raises(ValueError, match="tau must be strictly in"):
            tpqp.predict_premium(_make_test_X(2), tau=1.0)

    def test_tau_negative_raises(self):
        tpqp = self._build_tpqp()
        with pytest.raises(ValueError, match="tau must be strictly in"):
            tpqp.predict_premium(_make_test_X(2), tau=-0.1)

    def test_tau_above_one_raises(self):
        tpqp = self._build_tpqp()
        with pytest.raises(ValueError, match="tau must be strictly in"):
            tpqp.predict_premium(_make_test_X(2), tau=1.5)

    def test_gamma_below_zero_raises(self):
        tpqp = self._build_tpqp()
        with pytest.raises(ValueError, match="gamma must be in"):
            tpqp.predict_premium(_make_test_X(2), gamma=-0.01)

    def test_gamma_above_one_raises(self):
        tpqp = self._build_tpqp()
        with pytest.raises(ValueError, match="gamma must be in"):
            tpqp.predict_premium(_make_test_X(2), gamma=1.01)

    def test_unfitted_sev_model_raises(self):
        sev = QuantileGBM(quantiles=[0.5, 0.9])  # NOT fitted
        freq = _MockFreqModel(np.array([0.7]))
        tpqp = TwoPartQuantilePremium(freq, sev)
        with pytest.raises(RuntimeError, match="not fitted"):
            tpqp.predict_premium(_make_test_X(1))


# ---------------------------------------------------------------------------
# TestKnownAnswer: verify exact arithmetic from spec §4.1
# ---------------------------------------------------------------------------

class TestKnownAnswer:
    """
    Known-answer test using the worked example from the spec:

    Policy 1: p_i = 0.70, tau = 0.90, gamma = 0.50
      tau_i = (0.90 - 0.70) / 0.30 = 0.6667
      QuantileGBM levels [0.60, 0.70]: Q_0.60 = 800, Q_0.70 = 1000
      w = (0.6667 - 0.60) / 0.10 = 0.667
      sev_q = (1 - 0.667) * 800 + 0.667 * 1000 = 933.1
      mean_sev = 600 (mock)
      pure_premium = 0.30 * 600 = 180
      premium = 0.50 * 933.1 + 0.50 * 180 = 556.6
      safety_loading = 376.6

    Policy 2: p_i = 0.95, tau = 0.90 => p_i >= tau => FALLBACK
      premium = pure_premium = 0.05 * 600 = 30
      safety_loading = 0.0
    """

    def setup_method(self):
        n = 2
        p_no_claim = np.array([0.70, 0.95])
        freq = _MockFreqModel(p_no_claim)
        sev = _make_mock_sev_model([0.60, 0.70], np.array([800.0, 1000.0]), n)
        mean_model = _MockMeanSevModel(600.0)
        self.tpqp = TwoPartQuantilePremium(freq, sev, mean_sev_model=mean_model)
        self.X = _make_test_X(n)

    def test_policy1_premium(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert abs(float(result.premium[0]) - 556.55) < 1.0, (
            f"Expected ~556.6, got {result.premium[0]}"
        )

    def test_policy2_premium_fallback(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert abs(float(result.premium[1]) - 30.0) < 0.01, (
            f"Expected 30.0, got {result.premium[1]}"
        )

    def test_policy1_safety_loading(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert abs(float(result.safety_loading[0]) - 376.55) < 1.0, (
            f"Expected ~376.6, got {result.safety_loading[0]}"
        )

    def test_policy2_safety_loading_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert float(result.safety_loading[1]) == 0.0

    def test_n_fallback_is_one(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert result.n_fallback == 1

    def test_policy2_adjusted_tau_is_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert np.isnan(float(result.adjusted_tau[1]))

    def test_policy1_adjusted_tau(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        expected_tau_i = (0.90 - 0.70) / 0.30
        assert abs(float(result.adjusted_tau[0]) - expected_tau_i) < 1e-6

    def test_pure_premium_policy1(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert abs(float(result.pure_premium[0]) - 180.0) < 0.01

    def test_pure_premium_policy2(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert abs(float(result.pure_premium[1]) - 30.0) < 0.01

    def test_result_tau_and_gamma_stored(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert result.tau == 0.90
        assert result.gamma == 0.50


# ---------------------------------------------------------------------------
# TestAllFallback: edge case where every policy is a fallback
# ---------------------------------------------------------------------------

class TestAllFallback:
    """When all p_i >= tau, all premiums equal pure premiums."""

    def setup_method(self):
        n = 5
        p_no_claim = np.full(n, 0.95)  # all >= tau=0.90
        freq = _MockFreqModel(p_no_claim)
        sev = _make_mock_sev_model([0.60, 0.70], np.array([800.0, 1000.0]), n)
        mean_model = _MockMeanSevModel(600.0)
        self.tpqp = TwoPartQuantilePremium(freq, sev, mean_sev_model=mean_model)
        self.n = n
        self.X = _make_test_X(n)

    def test_n_fallback_equals_n(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert result.n_fallback == self.n

    def test_all_safety_loadings_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert (result.safety_loading.to_numpy() == 0.0).all()

    def test_premium_equals_pure_premium(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        np.testing.assert_array_almost_equal(
            result.premium.to_numpy(),
            result.pure_premium.to_numpy(),
        )

    def test_all_adjusted_tau_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)
        assert np.isnan(result.adjusted_tau.to_numpy()).all()

    def test_fallback_warning_issued(self):
        with pytest.warns(UserWarning, match="fallback"):
            self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.50)


# ---------------------------------------------------------------------------
# TestGammaBoundary: gamma = 0.0 and gamma = 1.0 edge cases
# ---------------------------------------------------------------------------

class TestGammaBoundary:
    def setup_method(self):
        n = 3
        p_no_claim = np.array([0.70, 0.60, 0.50])  # all valid at tau=0.90
        freq = _MockFreqModel(p_no_claim)
        sev = _make_mock_sev_model([0.60, 0.70], np.array([800.0, 1000.0]), n)
        mean_model = _MockMeanSevModel(600.0)
        self.tpqp = TwoPartQuantilePremium(freq, sev, mean_sev_model=mean_model)
        self.X = _make_test_X(n)

    def test_gamma_zero_premium_equals_pure_premium(self):
        result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.0)
        np.testing.assert_array_almost_equal(
            result.premium.to_numpy(),
            result.pure_premium.to_numpy(),
            decimal=6,
        )

    def test_gamma_one_premium_equals_sev_quantile(self):
        result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=1.0)
        sev_q = result.severity_quantile.to_numpy()
        prem = result.premium.to_numpy()
        # All policies are valid (no fallback), so premium should equal sev_q
        np.testing.assert_array_almost_equal(prem, sev_q, decimal=5)

    def test_gamma_zero_safety_loading_is_zero(self):
        result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.0)
        np.testing.assert_array_almost_equal(
            result.safety_loading.to_numpy(),
            np.zeros(3),
            decimal=6,
        )


# ---------------------------------------------------------------------------
# TestExtrapolationWarning: tau_i beyond QuantileGBM max quantile
# ---------------------------------------------------------------------------

class TestExtrapolationWarning:
    """
    When majority of tau_i exceed the QuantileGBM's highest quantile level,
    a UserWarning should be issued about extrapolation.

    Setup: sev_model only has quantile levels [0.5, 0.6].
    Set p_i very low (e.g. 0.01) so tau_i = (0.90 - 0.01) / 0.99 ~ 0.90,
    which exceeds the model's max of 0.6.
    """

    def setup_method(self):
        n = 10
        p_no_claim = np.full(n, 0.01)   # very low -> tau_i ~ 0.90 >> max q_level 0.6
        freq = _MockFreqModel(p_no_claim)
        sev = _make_mock_sev_model([0.5, 0.6], np.array([800.0, 1000.0]), n)
        mean_model = _MockMeanSevModel(600.0)
        self.tpqp = TwoPartQuantilePremium(freq, sev, mean_sev_model=mean_model)
        self.X = _make_test_X(n)

    def test_extrapolation_warning_issued(self):
        with pytest.warns(UserWarning, match="exceed the QuantileGBM"):
            self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.5)


# ---------------------------------------------------------------------------
# TestFallbackWarning: p_i >= tau emits UserWarning
# ---------------------------------------------------------------------------

class TestFallbackWarning:
    def test_fallback_warning_message(self):
        n = 3
        p_no_claim = np.array([0.92, 0.95, 0.98])  # all >= tau=0.90
        freq = _MockFreqModel(p_no_claim)
        sev = _make_mock_sev_model([0.60, 0.70], np.array([800.0, 1000.0]), n)
        tpqp = TwoPartQuantilePremium(freq, sev)
        X = _make_test_X(n)
        with pytest.warns(UserWarning, match="3 of 3 policies"):
            tpqp.predict_premium(X, tau=0.90, gamma=0.5)


# ---------------------------------------------------------------------------
# TestOutputTypes: verify Series names and types
# ---------------------------------------------------------------------------

class TestOutputTypes:
    def setup_method(self):
        n = 4
        p_no_claim = np.array([0.70, 0.60, 0.55, 0.65])
        freq = _MockFreqModel(p_no_claim)
        sev = _make_mock_sev_model([0.6, 0.7, 0.8], np.array([500.0, 700.0, 900.0]), n)
        mean_model = _MockMeanSevModel(400.0)
        self.tpqp = TwoPartQuantilePremium(freq, sev, mean_sev_model=mean_model)
        self.X = _make_test_X(n)
        self.result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.5)

    def test_premium_is_polars_series(self):
        assert isinstance(self.result.premium, pl.Series)

    def test_pure_premium_is_polars_series(self):
        assert isinstance(self.result.pure_premium, pl.Series)

    def test_safety_loading_is_polars_series(self):
        assert isinstance(self.result.safety_loading, pl.Series)

    def test_no_claim_prob_is_polars_series(self):
        assert isinstance(self.result.no_claim_prob, pl.Series)

    def test_adjusted_tau_is_polars_series(self):
        assert isinstance(self.result.adjusted_tau, pl.Series)

    def test_severity_quantile_is_polars_series(self):
        assert isinstance(self.result.severity_quantile, pl.Series)

    def test_n_fallback_is_int(self):
        assert isinstance(self.result.n_fallback, int)

    def test_correct_length(self):
        n = 4
        assert len(self.result.premium) == n
        assert len(self.result.pure_premium) == n
        assert len(self.result.safety_loading) == n

    def test_no_nulls_in_premium(self):
        assert self.result.premium.is_null().sum() == 0

    def test_premium_column_name(self):
        assert self.result.premium.name == "premium"

    def test_pure_premium_column_name(self):
        assert self.result.pure_premium.name == "pure_premium"

    def test_safety_loading_equals_premium_minus_pure(self):
        loading_computed = (
            self.result.premium.to_numpy() - self.result.pure_premium.to_numpy()
        )
        np.testing.assert_array_almost_equal(
            self.result.safety_loading.to_numpy(),
            loading_computed,
            decimal=10,
        )


# ---------------------------------------------------------------------------
# TestMeanSevFallback: predict without mean_sev_model (trapezoid approximation)
# ---------------------------------------------------------------------------

class TestMeanSevFallback:
    """Verify behaviour when mean_sev_model is None."""

    def setup_method(self):
        n = 3
        p_no_claim = np.array([0.70, 0.60, 0.55])
        freq = _MockFreqModel(p_no_claim)
        # Use 5 quantile levels for a more accurate trapezoid approximation
        sev = _make_mock_sev_model(
            [0.5, 0.6, 0.7, 0.8, 0.9],
            np.array([500.0, 600.0, 700.0, 800.0, 900.0]),
            n,
        )
        # No mean_sev_model
        self.tpqp = TwoPartQuantilePremium(freq, sev)
        self.X = _make_test_X(n)

    def test_runs_without_error(self):
        result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.5)
        assert len(result.premium) == 3

    def test_premiums_are_finite(self):
        result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.5)
        assert np.isfinite(result.premium.to_numpy()).all()

    def test_loading_non_negative_for_valid_policies(self):
        result = self.tpqp.predict_premium(self.X, tau=0.90, gamma=0.5)
        # All valid (no fallback), gamma=0.5 > 0 — loading should be >= 0
        # (true when sev_quantile >= pure_premium, which holds for quantiles above mean)
        assert (result.safety_loading.to_numpy() >= -0.01).all()


# ---------------------------------------------------------------------------
# TestTwoPartIntegration: full integration test with real models
# Spec §4.6 — fits LogisticRegression + QuantileGBM on synthetic data
# ---------------------------------------------------------------------------

class TestTwoPartIntegration:
    """
    Integration test: fit real models on synthetic zero-inflated data.

    This test is slower (fits CatBoost) but verifies the end-to-end workflow
    works correctly with real sklearn and QuantileGBM objects.
    """

    @pytest.fixture(scope="class")
    def fitted_result(self):
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(42)
        n = 2000  # smaller than spec for speed on CI
        X = pl.DataFrame({
            "age": rng.uniform(20, 80, n),
            "bm": rng.uniform(0, 5, n),
        })
        # 25% claim rate
        y_freq = (rng.uniform(size=n) < 0.25).astype(int)
        y_sev = np.where(y_freq == 1, rng.lognormal(7, 1.5, n), 0.0)

        # Frequency model on all rows
        freq_model = LogisticRegression(max_iter=500, random_state=42)
        freq_model.fit(X.to_numpy(), y_freq)

        # Severity quantile model on non-zero rows only
        mask = y_sev > 0
        X_sev = X.filter(pl.Series(mask))
        y_sev_pos = pl.Series("y_sev", y_sev[mask])
        sev_model = QuantileGBM(
            quantiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            iterations=200,
            learning_rate=0.1,
            depth=4,
        )
        sev_model.fit(X_sev, y_sev_pos)

        tpqp = TwoPartQuantilePremium(freq_model, sev_model)
        result = tpqp.predict_premium(X, tau=0.9, gamma=0.5)
        return result, n

    def test_premium_no_nulls(self, fitted_result):
        result, _ = fitted_result
        assert result.premium.is_null().sum() == 0

    def test_premium_length(self, fitted_result):
        result, n = fitted_result
        assert len(result.premium) == n

    def test_formula_consistency(self, fitted_result):
        """premium = gamma * sev_q + (1-gamma) * pure_premium for valid policies."""
        result, _ = fitted_result
        prem = result.premium.to_numpy()
        pure = result.pure_premium.to_numpy()
        sev_q = result.severity_quantile.to_numpy()
        tau_i = result.adjusted_tau.to_numpy()
        valid = ~np.isnan(tau_i)
        gamma = result.gamma
        expected = gamma * sev_q[valid] + (1 - gamma) * pure[valid]
        np.testing.assert_array_almost_equal(prem[valid], expected, decimal=5)

    def test_fallback_premium_equals_pure(self, fitted_result):
        """Fallback policies: premium exactly equals pure_premium."""
        result, _ = fitted_result
        tau_i = result.adjusted_tau.to_numpy()
        fallback = np.isnan(tau_i)
        if fallback.sum() > 0:
            np.testing.assert_array_almost_equal(
                result.premium.to_numpy()[fallback],
                result.pure_premium.to_numpy()[fallback],
                decimal=10,
            )

    def test_premiums_finite_and_positive(self, fitted_result):
        """All premiums should be finite. Most should be positive for insurance data."""
        result, _ = fitted_result
        prem = result.premium.to_numpy()
        assert np.isfinite(prem).all()
        assert (prem >= 0.0).all()

    def test_fallback_count_reasonable(self, fitted_result):
        result, n = fitted_result
        # With 25% claim rate, most p_i ~ 0.75 < tau=0.90, so fallbacks should be few
        assert result.n_fallback < n * 0.5

    def test_result_is_twopartresult(self, fitted_result):
        result, _ = fitted_result
        assert isinstance(result, TwoPartResult)

    def test_adjusted_tau_in_range(self, fitted_result):
        result, _ = fitted_result
        tau_arr = result.adjusted_tau.to_numpy()
        valid = tau_arr[~np.isnan(tau_arr)]
        assert ((valid > 0.0) & (valid < 1.0)).all()

    def test_no_claim_prob_in_range(self, fitted_result):
        result, _ = fitted_result
        p = result.no_claim_prob.to_numpy()
        assert ((p >= 0.0) & (p <= 1.0)).all()
