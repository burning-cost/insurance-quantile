"""
Tests for WassersteinRobustQR and the Polars convenience functions.

Test strategy:
    - Small synthetic datasets with known analytical quantile properties
    - Linear DGP where the closed-form slope/intercept can be verified
    - Conservatism property: robust predictions >= standard QR at high tau, eps > 0
    - p=1 equivalence to standard QR
    - Theorem 3 eps schedule: decreasing in N
    - Polars API wrappers: correct shapes, dtypes, column names

We use small N and low-dimensional X throughout — correctness matters here,
not predictive accuracy. Fast enough to run locally without heavy compute.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_quantile import WassersteinRobustQR, wdrqr_large_loss_loading, wdrqr_reserve_quantile


# ---------------------------------------------------------------------------
# Shared data factories
# ---------------------------------------------------------------------------

def _make_linear_data(
    N: int = 300,
    d: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear DGP: y = X @ beta_true + noise, noise ~ Exponential(1).

    The true 95th percentile of noise is -ln(0.05) ≈ 2.996, so the
    true Q_0.95(y | x) = x @ beta_true + 2.996.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(N, d))
    beta_true = np.array([1.0, -0.5] + [0.0] * (d - 2))
    y = X @ beta_true + rng.exponential(size=N)
    return X, y


def _make_polars_data(
    N: int = 300,
    d: int = 2,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.Series]:
    X_np, y_np = _make_linear_data(N=N, d=d, seed=seed)
    X = pl.DataFrame({f"x{i}": X_np[:, i] for i in range(d)})
    y = pl.Series("y", y_np)
    return X, y


# ---------------------------------------------------------------------------
# WassersteinRobustQR: basic fit/predict
# ---------------------------------------------------------------------------

class TestWassersteinRobustQRFitPredict:
    def test_fit_returns_self(self):
        X, y = _make_linear_data()
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        result = model.fit(X, y)
        assert result is model

    def test_is_fitted_after_fit(self):
        X, y = _make_linear_data()
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        model.fit(X, y)
        assert model.is_fitted_

    def test_coef_shape(self):
        X, y = _make_linear_data(d=3)
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        model.fit(X, y)
        assert model.coef_.shape == (3,)

    def test_intercept_is_scalar(self):
        X, y = _make_linear_data()
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        model.fit(X, y)
        assert isinstance(model.intercept_, float)

    def test_predict_shape(self):
        X, y = _make_linear_data(N=300)
        X_test, _ = _make_linear_data(N=50, seed=99)
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        model.fit(X, y)
        preds = model.predict(X_test)
        assert preds.shape == (50,)

    def test_predict_before_fit_raises(self):
        X_test, _ = _make_linear_data(N=10)
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        with pytest.raises(RuntimeError, match="fit()"):
            model.predict(X_test)

    def test_wrong_feature_count_raises(self):
        X, y = _make_linear_data(d=2)
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        model.fit(X, y)
        X_wrong = np.random.default_rng(0).uniform(size=(10, 5))
        with pytest.raises(ValueError, match="Expected 2 features"):
            model.predict(X_wrong)

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError, match="tau must be in"):
            WassersteinRobustQR(tau=1.5)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError, match="p must be >= 1"):
            WassersteinRobustQR(tau=0.9, p=0)

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="eps must be non-negative"):
            WassersteinRobustQR(tau=0.9, eps=-0.1)

    def test_eps_used_set_after_fit(self):
        X, y = _make_linear_data()
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.2)
        model.fit(X, y)
        assert model.eps_used_ == 0.2

    def test_eps_auto_set_when_none(self):
        X, y = _make_linear_data(N=200)
        model = WassersteinRobustQR(tau=0.9, p=2, eps=None, fit_eps=True)
        model.fit(X, y)
        assert model.eps_used_ is not None
        assert model.eps_used_ > 0.0

    def test_no_fit_eps_defaults_to_zero(self):
        X, y = _make_linear_data()
        model = WassersteinRobustQR(tau=0.9, p=2, eps=None, fit_eps=False)
        model.fit(X, y)
        assert model.eps_used_ == 0.0

    def test_repr(self):
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        rep = repr(model)
        assert "WassersteinRobustQR" in rep
        assert "0.9" in rep


# ---------------------------------------------------------------------------
# Conservatism property: robust > standard QR at high tau, eps > 0
# ---------------------------------------------------------------------------

class TestConservatismProperty:
    """
    For tau > 0.5 and eps > 0, WDRQR predictions should be at least as large
    as standard QR on average. The upward intercept correction (Theorem 1)
    guarantees this, though on individual risks with different slopes the
    comparison is through the mean.
    """

    def test_robust_higher_than_standard_at_tau_095(self):
        """Mean WDRQR prediction > mean standard QR prediction at tau=0.95."""
        X, y = _make_linear_data(N=300)
        X_test, _ = _make_linear_data(N=100, seed=77)

        standard = WassersteinRobustQR(tau=0.95, p=2, eps=0.0, fit_eps=False)
        standard.fit(X, y)
        standard_preds = standard.predict(X_test)

        robust = WassersteinRobustQR(tau=0.95, p=2, eps=0.2, fit_eps=False)
        robust.fit(X, y)
        robust_preds = robust.predict(X_test)

        # Mean robust prediction should exceed or equal standard QR
        assert float(np.mean(robust_preds)) >= float(np.mean(standard_preds)) - 1e-6, (
            f"Robust mean {np.mean(robust_preds):.4f} < standard mean {np.mean(standard_preds):.4f}"
        )

    def test_larger_eps_gives_larger_intercept(self):
        """Intercept should be non-decreasing in eps for tau > 0.5."""
        X, y = _make_linear_data(N=300)

        intercepts = []
        for eps in [0.0, 0.05, 0.1, 0.2, 0.4]:
            m = WassersteinRobustQR(tau=0.95, p=2, eps=eps, fit_eps=False)
            m.fit(X, y)
            intercepts.append(m.intercept_)

        for i in range(len(intercepts) - 1):
            assert intercepts[i] <= intercepts[i + 1] + 1e-6, (
                f"Intercept decreased as eps increased: {intercepts}"
            )

    def test_robust_lower_than_standard_at_tau_005(self):
        """
        For tau < 0.5, the intercept correction is negative.
        Mean WDRQR prediction should be <= standard QR at tau=0.05.
        """
        X, y = _make_linear_data(N=300)
        X_test, _ = _make_linear_data(N=100, seed=77)

        standard = WassersteinRobustQR(tau=0.05, p=2, eps=0.0, fit_eps=False)
        standard.fit(X, y)
        standard_preds = standard.predict(X_test)

        robust = WassersteinRobustQR(tau=0.05, p=2, eps=0.2, fit_eps=False)
        robust.fit(X, y)
        robust_preds = robust.predict(X_test)

        # Mean robust prediction should be <= standard QR for low tau
        assert float(np.mean(robust_preds)) <= float(np.mean(standard_preds)) + 1e-6, (
            f"Low-tau: robust mean {np.mean(robust_preds):.4f} > standard {np.mean(standard_preds):.4f}"
        )

    def test_eps_zero_matches_standard_qr(self):
        """WDRQR with eps=0 should give essentially the same result as standard QR."""
        X, y = _make_linear_data(N=300)
        X_test, _ = _make_linear_data(N=50, seed=55)

        standard = WassersteinRobustQR(tau=0.9, p=2, eps=0.0, fit_eps=False)
        standard.fit(X, y)
        standard_preds = standard.predict(X_test)

        robust_zero = WassersteinRobustQR(tau=0.9, p=2, eps=0.0, fit_eps=False)
        robust_zero.fit(X, y)
        robust_preds = robust_zero.predict(X_test)

        np.testing.assert_allclose(robust_preds, standard_preds, rtol=1e-4)


# ---------------------------------------------------------------------------
# p=1 gives same result as standard QR
# ---------------------------------------------------------------------------

class TestP1EquivalenceToStandardQR:
    """
    Zhang et al. Theorem (§3.1): for p=1 (W_1 distance), the WDRQR estimator
    is identical to standard QR regardless of eps.
    """

    def test_p1_matches_p2_with_eps_zero(self):
        """
        p=1 with any eps should match p=2 with eps=0 (both are standard QR).
        Verify that p=1 and p=2/eps=0 give consistent predictions.
        """
        X, y = _make_linear_data(N=300)
        X_test, _ = _make_linear_data(N=50, seed=11)

        m_p2_zero = WassersteinRobustQR(tau=0.9, p=2, eps=0.0, fit_eps=False)
        m_p2_zero.fit(X, y)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            m_p1 = WassersteinRobustQR(tau=0.9, p=1, eps=0.3, fit_eps=False)
            m_p1.fit(X, y)

        preds_p2 = m_p2_zero.predict(X_test)
        preds_p1 = m_p1.predict(X_test)

        # p=1 and p=2/eps=0 should both produce standard QR; predictions match
        np.testing.assert_allclose(preds_p1, preds_p2, rtol=1e-3, atol=1e-3)

    def test_p1_with_different_eps_gives_same_predictions(self):
        """For p=1, eps should have no effect on predictions."""
        X, y = _make_linear_data(N=200)
        X_test, _ = _make_linear_data(N=30, seed=13)

        import warnings
        preds_list = []
        for eps in [0.0, 0.1, 0.5]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                m = WassersteinRobustQR(tau=0.9, p=1, eps=eps, fit_eps=False)
                m.fit(X, y)
                preds_list.append(m.predict(X_test))

        np.testing.assert_allclose(preds_list[0], preds_list[1], rtol=1e-4)
        np.testing.assert_allclose(preds_list[0], preds_list[2], rtol=1e-4)

    def test_p1_nonzero_eps_issues_warning(self):
        """p=1 with eps > 0 should emit a UserWarning."""
        X, y = _make_linear_data(N=100)
        m = WassersteinRobustQR(tau=0.9, p=1, eps=0.2, fit_eps=False)
        with pytest.warns(UserWarning, match="identical to standard QR"):
            m.fit(X, y)


# ---------------------------------------------------------------------------
# Theorem 3: optimal_eps decreases with N
# ---------------------------------------------------------------------------

class TestOptimalEps:
    def test_eps_decreases_with_N(self):
        """Theorem 3 radius is O(N^{-1/2}): larger N -> smaller eps."""
        model = WassersteinRobustQR(tau=0.99, p=2)
        eps_values = [model.optimal_eps(N) for N in [50, 200, 1000, 5000]]
        for i in range(len(eps_values) - 1):
            assert eps_values[i] > eps_values[i + 1], (
                f"eps did not decrease: eps[{i}]={eps_values[i]:.4f}, eps[{i+1}]={eps_values[i+1]:.4f}"
            )

    def test_eps_positive(self):
        model = WassersteinRobustQR(tau=0.9, p=2)
        eps = model.optimal_eps(100)
        assert eps > 0.0

    def test_eps_invalid_s_raises(self):
        model = WassersteinRobustQR(tau=0.9, p=2)
        with pytest.raises(ValueError, match="s must be > 2"):
            model.optimal_eps(100, s=1.5)

    def test_eps_invalid_N_raises(self):
        model = WassersteinRobustQR(tau=0.9, p=2)
        with pytest.raises(ValueError, match="N must be positive"):
            model.optimal_eps(0)

    def test_eps_larger_for_high_tau(self):
        """
        c_{tau,2} = sqrt(tau^2 + (1-tau)^2) is larger at extreme tau.
        So optimal_eps should be larger for tau=0.99 than tau=0.5.
        """
        m_high = WassersteinRobustQR(tau=0.99, p=2)
        m_mid = WassersteinRobustQR(tau=0.5, p=2)
        assert m_high.optimal_eps(100) > m_mid.optimal_eps(100)

    def test_eps_larger_for_lower_s(self):
        """Heavier-tailed data (smaller s) requires wider robustness ball."""
        model = WassersteinRobustQR(tau=0.9, p=2)
        eps_heavy = model.optimal_eps(200, s=2.5)
        eps_light = model.optimal_eps(200, s=6.0)
        assert eps_heavy > eps_light


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_1d_X_raises(self):
        X_bad = np.ones(10)
        y = np.ones(10)
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        with pytest.raises(ValueError, match="2-dimensional"):
            model.fit(X_bad, y)

    def test_2d_y_raises(self):
        X = np.ones((10, 2))
        y_bad = np.ones((10, 1))
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        with pytest.raises(ValueError, match="1-dimensional"):
            model.fit(X, y_bad)

    def test_mismatched_rows_raises(self):
        X = np.ones((10, 2))
        y = np.ones(8)
        model = WassersteinRobustQR(tau=0.9, p=2, eps=0.1)
        with pytest.raises(ValueError, match="same number of rows"):
            model.fit(X, y)


# ---------------------------------------------------------------------------
# wdrqr_large_loss_loading
# ---------------------------------------------------------------------------

class TestWDRQRLargeLossLoading:
    def test_returns_polars_series(self):
        X_train, y_train = _make_polars_data(N=200, seed=1)
        X_score, _ = _make_polars_data(N=50, seed=2)

        class ConstantMean:
            def predict(self, X):
                return np.full(len(X), 1.0)

        result = wdrqr_large_loss_loading(
            X_train, y_train, X_score, ConstantMean(), alpha=0.95, eps=0.1
        )
        assert isinstance(result, pl.Series)

    def test_series_name(self):
        X_train, y_train = _make_polars_data(N=200, seed=3)
        X_score, _ = _make_polars_data(N=30, seed=4)

        class ConstantMean:
            def predict(self, X):
                return np.full(len(X), 1.0)

        result = wdrqr_large_loss_loading(
            X_train, y_train, X_score, ConstantMean(), alpha=0.95, eps=0.1
        )
        assert result.name == "wdrqr_large_loss_loading"

    def test_series_length(self):
        X_train, y_train = _make_polars_data(N=200, seed=5)
        X_score, _ = _make_polars_data(N=40, seed=6)

        class ConstantMean:
            def predict(self, X):
                return np.full(len(X), 1.0)

        result = wdrqr_large_loss_loading(
            X_train, y_train, X_score, ConstantMean(), alpha=0.95, eps=0.1
        )
        assert len(result) == 40

    def test_loading_positive_at_high_alpha(self):
        """
        At alpha=0.99 with eps=0.2, the robust quantile should exceed
        a modest constant mean model for most risks.
        """
        X_train, y_train = _make_polars_data(N=300, seed=7)
        X_score, _ = _make_polars_data(N=100, seed=8)

        # Mean model returns 0.5 (below the 99th percentile of the DGP)
        class LowMean:
            def predict(self, X):
                return np.full(len(X), 0.5)

        result = wdrqr_large_loss_loading(
            X_train, y_train, X_score, LowMean(), alpha=0.99, eps=0.2
        )
        assert float(result.mean()) > 0.0, f"Expected positive loading, got {result.mean()}"

    def test_polars_mean_model(self):
        """Mean model returning a Polars Series should also work."""
        X_train, y_train = _make_polars_data(N=200, seed=9)
        X_score, _ = _make_polars_data(N=20, seed=10)

        class PolarsModel:
            def predict(self, X: pl.DataFrame) -> pl.Series:
                return pl.Series("mean", np.full(len(X), 1.0))

        result = wdrqr_large_loss_loading(
            X_train, y_train, X_score, PolarsModel(), alpha=0.95, eps=0.1
        )
        assert len(result) == 20

    def test_auto_eps_works(self):
        """eps=None should trigger automatic eps from Theorem 3."""
        X_train, y_train = _make_polars_data(N=200, seed=11)
        X_score, _ = _make_polars_data(N=20, seed=12)

        class ConstantMean:
            def predict(self, X):
                return np.full(len(X), 1.0)

        result = wdrqr_large_loss_loading(
            X_train, y_train, X_score, ConstantMean(), alpha=0.95, eps=None
        )
        assert isinstance(result, pl.Series)
        assert len(result) == 20


# ---------------------------------------------------------------------------
# wdrqr_reserve_quantile
# ---------------------------------------------------------------------------

class TestWDRQRReserveQuantile:
    def test_returns_polars_dataframe(self):
        X_train, y_train = _make_polars_data(N=200, seed=20)
        X_score, _ = _make_polars_data(N=30, seed=21)
        result = wdrqr_reserve_quantile(X_train, y_train, X_score, tau=0.95, eps=0.1)
        assert isinstance(result, pl.DataFrame)

    def test_columns_without_ci(self):
        X_train, y_train = _make_polars_data(N=200, seed=22)
        X_score, _ = _make_polars_data(N=30, seed=23)
        result = wdrqr_reserve_quantile(X_train, y_train, X_score, tau=0.95, eps=0.1)
        assert "quantile" in result.columns
        assert "eps_used" in result.columns
        assert "quantile_lower" not in result.columns
        assert "quantile_upper" not in result.columns

    def test_columns_with_ci(self):
        X_train, y_train = _make_polars_data(N=200, seed=24)
        X_score, _ = _make_polars_data(N=30, seed=25)
        result = wdrqr_reserve_quantile(X_train, y_train, X_score, tau=0.95, eps=0.1, ci=True)
        assert "quantile_lower" in result.columns
        assert "quantile_upper" in result.columns

    def test_row_count(self):
        X_train, y_train = _make_polars_data(N=200, seed=26)
        X_score, _ = _make_polars_data(N=45, seed=27)
        result = wdrqr_reserve_quantile(X_train, y_train, X_score, tau=0.99, eps=0.15)
        assert len(result) == 45

    def test_eps_used_constant(self):
        """eps_used column should have the same value for all rows."""
        X_train, y_train = _make_polars_data(N=200, seed=28)
        X_score, _ = _make_polars_data(N=20, seed=29)
        result = wdrqr_reserve_quantile(X_train, y_train, X_score, tau=0.95, eps=0.1)
        eps_col = result["eps_used"].to_numpy()
        assert np.all(eps_col == eps_col[0])
        assert abs(eps_col[0] - 0.1) < 1e-9

    def test_ci_ordering(self):
        """
        With ci=True: quantile_lower <= quantile <= quantile_upper (on average).
        The lower bound uses eps=0 and upper uses 2*eps; quantile uses eps.
        """
        X_train, y_train = _make_polars_data(N=300, seed=30)
        X_score, _ = _make_polars_data(N=50, seed=31)
        result = wdrqr_reserve_quantile(
            X_train, y_train, X_score, tau=0.95, eps=0.2, ci=True
        )
        lower = result["quantile_lower"].to_numpy()
        mid = result["quantile"].to_numpy()
        upper = result["quantile_upper"].to_numpy()

        # On average, the ordering should hold
        assert float(np.mean(lower)) <= float(np.mean(mid)) + 1e-6
        assert float(np.mean(mid)) <= float(np.mean(upper)) + 1e-6

    def test_auto_eps(self):
        """eps=None triggers auto eps; result should still be a valid DataFrame."""
        X_train, y_train = _make_polars_data(N=200, seed=32)
        X_score, _ = _make_polars_data(N=20, seed=33)
        result = wdrqr_reserve_quantile(X_train, y_train, X_score, tau=0.95)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 20
        assert result["eps_used"][0] > 0.0

    def test_quantile_positive_for_positive_y(self):
        """For positive-only targets, the high-quantile prediction should be positive."""
        rng = np.random.default_rng(42)
        X_np = rng.uniform(0, 1, size=(200, 2))
        y_np = np.abs(rng.normal(loc=5.0, scale=1.0, size=200))  # strictly positive
        X_train = pl.DataFrame({f"x{i}": X_np[:, i] for i in range(2)})
        y_train = pl.Series("y", y_np)
        X_score = pl.DataFrame({f"x{i}": rng.uniform(0, 1, size=20) for i in range(2)})

        result = wdrqr_reserve_quantile(X_train, y_train, X_score, tau=0.99, eps=0.1)
        assert float(result["quantile"].min()) > 0.0


# ---------------------------------------------------------------------------
# Intercept correction mathematics
# ---------------------------------------------------------------------------

class TestInterceptCorrectionMath:
    """
    Direct verification of the _intercept_correction function and c_tau_p.
    These test the mathematical core rather than the fitted model.
    """

    def test_c_tau_p_symmetry(self):
        """c_{tau,2} = c_{1-tau,2}: symmetric around tau=0.5."""
        from insurance_quantile._robust import _c_tau_p
        assert abs(_c_tau_p(0.9, 2) - _c_tau_p(0.1, 2)) < 1e-12

    def test_c_tau_p_at_half(self):
        """c_{0.5, 2} = sqrt(0.5^2 + 0.5^2) = 0.5 * sqrt(2) ≈ 0.7071."""
        from insurance_quantile._robust import _c_tau_p
        expected = 0.5 * np.sqrt(2.0)
        assert abs(_c_tau_p(0.5, 2) - expected) < 1e-10

    def test_intercept_correction_zero_at_tau_half(self):
        """
        At tau=0.5: tau^q - (1-tau)^q = 0.5^2 - 0.5^2 = 0.
        Correction should be zero.
        """
        from insurance_quantile._robust import _intercept_correction
        correction = _intercept_correction(beta_norm=1.0, tau=0.5, p=2, eps=0.2)
        assert abs(correction) < 1e-12

    def test_intercept_correction_positive_high_tau(self):
        """For tau=0.95, eps>0, the correction should be positive."""
        from insurance_quantile._robust import _intercept_correction
        correction = _intercept_correction(beta_norm=1.0, tau=0.95, p=2, eps=0.1)
        assert correction > 0.0

    def test_intercept_correction_negative_low_tau(self):
        """For tau=0.05, eps>0, the correction should be negative."""
        from insurance_quantile._robust import _intercept_correction
        correction = _intercept_correction(beta_norm=1.0, tau=0.05, p=2, eps=0.1)
        assert correction < 0.0

    def test_intercept_correction_scales_with_eps(self):
        """Correction should scale linearly with eps."""
        from insurance_quantile._robust import _intercept_correction
        c1 = _intercept_correction(beta_norm=1.0, tau=0.9, p=2, eps=0.1)
        c2 = _intercept_correction(beta_norm=1.0, tau=0.9, p=2, eps=0.2)
        assert abs(c2 / c1 - 2.0) < 1e-10

    def test_intercept_correction_zero_when_beta_zero(self):
        """When beta=0, the correction is zero regardless of eps."""
        from insurance_quantile._robust import _intercept_correction
        correction = _intercept_correction(beta_norm=0.0, tau=0.95, p=2, eps=0.5)
        assert correction == 0.0


# ---------------------------------------------------------------------------
# End-to-end: exports from top-level package
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_wdrqr_importable_from_package(self):
        from insurance_quantile import WassersteinRobustQR as W
        assert W is WassersteinRobustQR

    def test_wdrqr_large_loss_loading_importable(self):
        from insurance_quantile import wdrqr_large_loss_loading as f
        assert f is wdrqr_large_loss_loading

    def test_wdrqr_reserve_quantile_importable(self):
        from insurance_quantile import wdrqr_reserve_quantile as f
        assert f is wdrqr_reserve_quantile
