"""
Tests for ExpectedShortfallRegressor (i-Rock direct ES regression).

Test strategy:
    - Basic fit/predict on simulated data with known linear ES
    - summary() produces valid DataFrame with the correct columns
    - Coefficients recover true values on large samples (regression test,
      loose tolerance given the approximate two-stage procedure)
    - alpha parameter works at different levels
    - Edge cases: single feature, constant-ish response
    - Input validation: wrong shapes, bad parameters, predict before fit
    - Package-level import

All tests use small-to-moderate N (100-1000) and are designed to run fast
on serverless compute. No CatBoost or GPU dependencies.

Data generating process (DGP)
------------------------------
We use a linear DGP where the true ES is analytically tractable:

    Y | X ~ Q(alpha, X) + Exponential(1) / (1 - alpha)

This means ES(alpha, X) = Q(alpha, X) + 1/(1-alpha) * E[Exp(1)]
                         = Q(alpha, X) + 1/(1-alpha)

With Q(alpha, X) = x @ beta_true + (-log(1-alpha)), so:

    ES(alpha, X) = x @ beta_true + (-log(1-alpha)) + 1/(1-alpha)

The true ES slope is beta_true and the true intercept includes the
quantile intercept plus the expected tail excess.

Practically, for alpha=0.9, -log(0.1) ≈ 2.303 and 1/0.1 = 10.
For alpha=0.95, -log(0.05) ≈ 2.996 and 1/0.05 = 20.

We use a simpler location-shift DGP that gives easily interpretable results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_quantile import ExpectedShortfallRegressor


# ---------------------------------------------------------------------------
# Shared DGP helpers
# ---------------------------------------------------------------------------


def _make_linear_es_data(
    N: int = 500,
    p: int = 2,
    seed: int = 42,
    alpha: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Linear ES DGP: Y = X @ beta_true + noise, noise ~ Exponential(1).

    The true ES(alpha | X) for Exponential(1) tail is:
        ES(alpha, x) = x @ beta_true + (-log(1 - alpha)) + 1/(1 - alpha)

    Returns (X, y, beta_true, true_intercept).

    For alpha=0.9: true_intercept = -log(0.1) + 10 ≈ 12.303
    For alpha=0.95: true_intercept = -log(0.05) + 20 ≈ 22.996
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(N, p))
    beta_true = np.zeros(p)
    beta_true[0] = 1.0
    if p >= 2:
        beta_true[1] = -0.5
    noise = rng.exponential(size=N)
    y = X @ beta_true + noise
    true_intercept = -np.log(1.0 - alpha) + 1.0 / (1.0 - alpha)
    return X, y, beta_true, true_intercept


def _make_constant_response(
    N: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Near-constant response with a single feature.
    Used to test edge-case stability.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(N, 1))
    y = np.full(N, 2.0) + rng.normal(0, 0.01, size=N)
    return X, y


# ---------------------------------------------------------------------------
# Basic fit / predict
# ---------------------------------------------------------------------------


class TestExpectedShortfallRegressorFitPredict:
    def test_fit_returns_self(self):
        X, y, _, _ = _make_linear_es_data(N=200)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        result = model.fit(X, y)
        assert result is model

    def test_is_fitted_after_fit(self):
        X, y, _, _ = _make_linear_es_data(N=200)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        assert model.is_fitted_

    def test_coef_shape(self):
        X, y, _, _ = _make_linear_es_data(N=200, p=3)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        assert model.coef_.shape == (3,)

    def test_intercept_is_scalar(self):
        X, y, _, _ = _make_linear_es_data(N=200)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        assert isinstance(model.intercept_, float)

    def test_predict_shape(self):
        X, y, _, _ = _make_linear_es_data(N=300)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        preds = model.predict(X[:50])
        assert preds.shape == (50,)

    def test_predict_1d_input(self):
        """Single observation as 1D array should not raise."""
        X, y, _, _ = _make_linear_es_data(N=200)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        preds = model.predict(X[0])
        assert preds.shape == (1,)

    def test_predict_before_fit_raises(self):
        X, _, _, _ = _make_linear_es_data(N=50)
        model = ExpectedShortfallRegressor(alpha=0.9)
        with pytest.raises(RuntimeError, match="fit()"):
            model.predict(X)

    def test_summary_before_fit_raises(self):
        model = ExpectedShortfallRegressor(alpha=0.9)
        with pytest.raises(RuntimeError, match="fit()"):
            model.summary()

    def test_wrong_feature_count_raises(self):
        X, y, _, _ = _make_linear_es_data(N=200, p=2)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        X_bad = np.random.default_rng(0).uniform(size=(10, 5))
        with pytest.raises(ValueError, match="Expected 2 features"):
            model.predict(X_bad)

    def test_repr(self):
        model = ExpectedShortfallRegressor(alpha=0.95)
        rep = repr(model)
        assert "ExpectedShortfallRegressor" in rep
        assert "0.95" in rep

    def test_predict_positive_for_positive_y(self):
        """ES predictions should be positive when all y > 0."""
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, size=(300, 2))
        y = np.abs(rng.normal(5, 1, size=300))
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert float(np.min(preds)) > 0.0, (
            f"Expected all positive ES predictions, got min={np.min(preds):.4f}"
        )


# ---------------------------------------------------------------------------
# summary() structure
# ---------------------------------------------------------------------------


class TestSummaryDataFrame:
    def test_summary_returns_dataframe(self):
        X, y, _, _ = _make_linear_es_data(N=300)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        summ = model.summary()
        assert isinstance(summ, pd.DataFrame)

    def test_summary_columns(self):
        X, y, _, _ = _make_linear_es_data(N=300)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        summ = model.summary()
        assert set(summ.columns) == {"term", "coef", "std_err", "z_stat", "p_value"}

    def test_summary_row_count(self):
        """p features + intercept = p + 1 rows."""
        X, y, _, _ = _make_linear_es_data(N=300, p=3)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        summ = model.summary()
        assert len(summ) == 4  # intercept + 3 features

    def test_summary_intercept_row(self):
        X, y, _, _ = _make_linear_es_data(N=300)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        summ = model.summary()
        assert "(Intercept)" in summ["term"].values

    def test_summary_coef_matches_coef_(self):
        """summary() coefficient column should match model.coef_ and model.intercept_."""
        X, y, _, _ = _make_linear_es_data(N=300, p=2)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        summ = model.summary()
        # Row 0: intercept
        np.testing.assert_allclose(
            summ["coef"].iloc[0], model.intercept_, rtol=1e-10
        )
        # Rows 1+: slope
        np.testing.assert_allclose(
            summ["coef"].iloc[1:].values, model.coef_, rtol=1e-10
        )

    def test_summary_std_err_positive(self):
        """Standard errors should be non-negative."""
        X, y, _, _ = _make_linear_es_data(N=400)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=4)
        model.fit(X, y)
        summ = model.summary()
        assert (summ["std_err"] >= 0).all()

    def test_summary_p_values_in_range(self):
        """p-values should lie in [0, 1]."""
        X, y, _, _ = _make_linear_es_data(N=400)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=4)
        model.fit(X, y)
        summ = model.summary()
        valid = summ["p_value"].notna()
        assert (summ.loc[valid, "p_value"] >= 0.0).all()
        assert (summ.loc[valid, "p_value"] <= 1.0).all()

    def test_summary_z_stat_consistent(self):
        """z_stat should equal coef / std_err (where std_err > 0)."""
        X, y, _, _ = _make_linear_es_data(N=400)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=4)
        model.fit(X, y)
        summ = model.summary()
        se_positive = summ["std_err"] > 1e-10
        if se_positive.any():
            np.testing.assert_allclose(
                summ.loc[se_positive, "z_stat"].values,
                (summ.loc[se_positive, "coef"] / summ.loc[se_positive, "std_err"]).values,
                rtol=1e-6,
            )

    def test_single_feature_summary(self):
        """Single-feature model should produce a 2-row summary."""
        X, y, _, _ = _make_linear_es_data(N=300, p=1)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        summ = model.summary()
        assert len(summ) == 2  # intercept + x0


# ---------------------------------------------------------------------------
# Coefficient recovery on large samples
# ---------------------------------------------------------------------------


class TestCoefficientRecovery:
    """
    On large samples the i-Rock estimator should recover the true ES slope.

    We use a loose tolerance (absolute error < 0.5) because:
    1. The two-stage procedure introduces non-trivial variance.
    2. The first-stage linear QR is approximate on finite samples.
    3. We are testing direction and order-of-magnitude, not precision.
    """

    def test_slope_sign_correct_large_sample(self):
        """
        The sign of the first coefficient should match the true DGP slope.
        True beta_true[0] = 1.0, so coef_[0] should be positive.
        """
        X, y, beta_true, _ = _make_linear_es_data(N=1000, p=2, seed=0)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=5)
        model.fit(X, y)
        assert model.coef_[0] > 0.0, (
            f"Expected coef_[0] > 0, got {model.coef_[0]:.4f}"
        )

    def test_second_slope_sign_correct(self):
        """
        True beta_true[1] = -0.5, so coef_[1] should be negative.
        """
        X, y, beta_true, _ = _make_linear_es_data(N=1000, p=2, seed=1)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=5)
        model.fit(X, y)
        assert model.coef_[1] < 0.0, (
            f"Expected coef_[1] < 0, got {model.coef_[1]:.4f}"
        )

    def test_intercept_in_plausible_range(self):
        """
        True intercept for alpha=0.9, Exp(1) noise:
            -log(0.1) + 1/0.1 ≈ 2.303 + 10 = 12.303

        The i-Rock estimate should be in a reasonable neighbourhood.
        """
        X, y, _, true_intercept = _make_linear_es_data(N=1000, p=2, seed=2, alpha=0.9)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=5)
        model.fit(X, y)
        # Loose tolerance: within 3 units of truth
        assert abs(model.intercept_ - true_intercept) < 3.0, (
            f"Intercept {model.intercept_:.3f} too far from truth {true_intercept:.3f}"
        )

    def test_predictions_exceed_quantile(self):
        """
        By definition, ES >= VaR at the same level.
        Mean ES prediction should exceed the sample alpha-quantile of y.
        """
        X, y, _, _ = _make_linear_es_data(N=800, p=2, seed=3)
        alpha = 0.9
        model = ExpectedShortfallRegressor(alpha=alpha, n_bins_per_dim=4)
        model.fit(X, y)
        preds = model.predict(X)
        sample_var = float(np.quantile(y, alpha))
        mean_es_pred = float(np.mean(preds))
        assert mean_es_pred > sample_var, (
            f"Mean ES {mean_es_pred:.4f} should exceed VaR {sample_var:.4f}"
        )


# ---------------------------------------------------------------------------
# Different alpha levels
# ---------------------------------------------------------------------------


class TestAlphaParameter:
    def test_higher_alpha_gives_larger_es(self):
        """
        ES is monotone in alpha: ES(0.99) >= ES(0.95) for the same X.
        Test on the mean predictions.
        """
        X, y, _, _ = _make_linear_es_data(N=800, p=2, seed=10)

        model_95 = ExpectedShortfallRegressor(alpha=0.95, n_bins_per_dim=4)
        model_95.fit(X, y)
        preds_95 = model_95.predict(X)

        model_99 = ExpectedShortfallRegressor(alpha=0.99, n_bins_per_dim=3)
        model_99.fit(X, y)
        preds_99 = model_99.predict(X)

        assert float(np.mean(preds_99)) > float(np.mean(preds_95)), (
            f"Expected mean ES(0.99) > mean ES(0.95), got "
            f"{np.mean(preds_99):.4f} vs {np.mean(preds_95):.4f}"
        )

    def test_fit_at_alpha_075(self):
        """Basic smoke test at a low alpha — should run without error."""
        X, y, _, _ = _make_linear_es_data(N=300, seed=11)
        model = ExpectedShortfallRegressor(alpha=0.75, n_bins_per_dim=3)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    def test_fit_at_alpha_095(self):
        """Smoke test at alpha=0.95."""
        X, y, _, _ = _make_linear_es_data(N=300, seed=12, alpha=0.95)
        model = ExpectedShortfallRegressor(alpha=0.95, n_bins_per_dim=3)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_feature(self):
        """Single-feature model should fit and predict cleanly."""
        X, y, _, _ = _make_linear_es_data(N=300, p=1, seed=20)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=4)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (300,)
        assert model.coef_.shape == (1,)

    def test_near_constant_response(self):
        """Near-constant y should not raise; predictions should be finite."""
        X, y = _make_constant_response(N=200)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds)), "ES predictions contain non-finite values"

    def test_large_n_bins_per_dim(self):
        """Large n_bins_per_dim should run without error (sparse bins handled)."""
        X, y, _, _ = _make_linear_es_data(N=500, p=1, seed=30)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=8)
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_auto_bins(self):
        """Default n_bins_per_dim='auto' should run without error."""
        X, y, _, _ = _make_linear_es_data(N=400, p=2, seed=31)
        model = ExpectedShortfallRegressor(alpha=0.9)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    def test_many_features(self):
        """More features (p=4) should work with appropriate bin count."""
        X, y, _, _ = _make_linear_es_data(N=600, p=4, seed=32)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=2)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_invalid_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            ExpectedShortfallRegressor(alpha=0.0)

    def test_invalid_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            ExpectedShortfallRegressor(alpha=1.0)

    def test_invalid_alpha_above_one_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            ExpectedShortfallRegressor(alpha=1.5)

    def test_invalid_n_quantile_grid_raises(self):
        with pytest.raises(ValueError, match="n_quantile_grid must be >= 1"):
            ExpectedShortfallRegressor(alpha=0.9, n_quantile_grid=0)

    def test_invalid_first_stage_raises(self):
        with pytest.raises(ValueError, match="first_stage must be"):
            ExpectedShortfallRegressor(alpha=0.9, first_stage="xgboost")

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon must be in"):
            ExpectedShortfallRegressor(alpha=0.9, epsilon=0.15)  # 1-alpha = 0.1, too large

    def test_1d_X_raises(self):
        X_bad = np.ones(10)
        y = np.ones(10)
        model = ExpectedShortfallRegressor(alpha=0.9)
        with pytest.raises(ValueError, match="2-dimensional"):
            model.fit(X_bad, y)

    def test_2d_y_raises(self):
        X = np.ones((10, 2))
        y_bad = np.ones((10, 1))
        model = ExpectedShortfallRegressor(alpha=0.9)
        with pytest.raises(ValueError, match="1-dimensional"):
            model.fit(X, y_bad)

    def test_mismatched_rows_raises(self):
        X = np.ones((10, 2))
        y = np.ones(8)
        model = ExpectedShortfallRegressor(alpha=0.9)
        with pytest.raises(ValueError, match="same number of rows"):
            model.fit(X, y)

    def test_n_bins_per_dim_less_than_2_raises(self):
        X, y, _, _ = _make_linear_es_data(N=200)
        model = ExpectedShortfallRegressor(alpha=0.9, n_bins_per_dim=1)
        with pytest.raises(ValueError, match="n_bins_per_dim must be >= 2"):
            model.fit(X, y)


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_importable_from_package(self):
        from insurance_quantile import ExpectedShortfallRegressor as ESR
        assert ESR is ExpectedShortfallRegressor

    def test_in_all(self):
        import insurance_quantile
        assert "ExpectedShortfallRegressor" in insurance_quantile.__all__
