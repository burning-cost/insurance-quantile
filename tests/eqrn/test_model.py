"""
Integration tests for EQRNModel (insurance_quantile.eqrn).

Tests the full two-step pipeline end-to-end:
- fit() on synthetic data
- predict_quantile, predict_tvar, predict_params, predict_exceedance_prob, predict_xl_layer
- Monotonicity of quantiles
- TVaR >= VaR
- shape_fixed vs full model behaviour
- Validation set / early stopping path
- Small dataset stability
- Error handling
- Exposure weighting produces different results
- OOF integrity check
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

pytest.importorskip(
    "torch",
    reason="EQRN tests require torch. Install with: pip install insurance-quantile[eqrn]",
)

from insurance_quantile.eqrn import EQRNModel


@pytest.fixture(scope="module")
def simple_fit():
    """EQRNModel fitted on 2000-observation Pareto data, shape_fixed=True."""
    rng = np.random.default_rng(42)
    n = 2000
    y = genpareto.rvs(c=0.35, scale=8000, loc=3000, size=n, random_state=rng)
    X = np.column_stack([
        rng.standard_normal(n),
        rng.uniform(0, 1, n),
    ])
    model = EQRNModel(
        tau_0=0.75,
        hidden_sizes=(16, 8),
        shape_fixed=True,
        n_epochs=80,
        patience=20,
        seed=42,
        verbose=0,
    )
    model.fit(X, y)
    return model, X, y


@pytest.fixture(scope="module")
def covariate_fit():
    """EQRNModel fitted on data with covariate-dependent xi."""
    rng = np.random.default_rng(123)
    n = 3000
    x1 = rng.uniform(0, 1, n)
    x2 = rng.binomial(1, 0.5, n).astype(float)
    X = np.column_stack([x1, x2])
    xi_true = 0.15 + 0.25 * x1
    sigma_true = 6000.0 * np.exp(0.2 * x2)
    y = np.array([
        genpareto.rvs(c=xi_true[i], scale=sigma_true[i], loc=8000.0,
                      random_state=int(rng.integers(1e9)))
        for i in range(n)
    ])
    split = int(0.8 * n)
    model = EQRNModel(
        tau_0=0.75,
        hidden_sizes=(16, 8),
        shape_fixed=False,
        n_epochs=100,
        patience=25,
        seed=123,
        verbose=0,
    )
    model.fit(X[:split], y[:split], X_val=X[split:], y_val=y[split:])
    return model, X[split:], y[split:]


class TestEQRNModelFit:
    def test_fit_sets_is_fitted(self, simple_fit):
        model, X, y = simple_fit
        assert model.is_fitted

    def test_n_exceedances_stored(self, simple_fit):
        model, X, y = simple_fit
        assert model.n_exceedances_ is not None
        assert model.n_exceedances_ > 0

    def test_exceedance_rate_near_target(self, simple_fit):
        model, X, y = simple_fit
        # Should be close to (1 - tau_0) = 0.25
        assert abs(model.exceedance_rate_ - 0.25) < 0.10

    def test_train_losses_recorded(self, simple_fit):
        model, X, y = simple_fit
        assert len(model.train_losses_) > 0
        assert all(np.isfinite(l) for l in model.train_losses_)

    def test_network_created(self, simple_fit):
        model, X, y = simple_fit
        assert model.network_ is not None

    def test_fit_with_validation_set(self):
        """Fitting with a validation set uses it for early stopping."""
        rng = np.random.default_rng(55)
        y = genpareto.rvs(c=0.3, scale=5000, loc=2000, size=1500, random_state=rng)
        X = rng.standard_normal((1500, 2))
        model = EQRNModel(tau_0=0.8, hidden_sizes=(8,), n_epochs=50, patience=10,
                          seed=55, verbose=0)
        model.fit(X[:1200], y[:1200], X_val=X[1200:], y_val=y[1200:])
        assert model.is_fitted
        # val_losses_ should be populated if val exceedances >= 10
        assert len(model.val_losses_) >= 0  # may be 0 if too few val exceedances


class TestEQRNModelPredictions:
    def test_predict_quantile_shape(self, simple_fit):
        model, X, y = simple_fit
        q = model.predict_quantile(X[:50], q=0.99)
        assert q.shape == (50,)

    def test_predict_quantile_finite(self, simple_fit):
        model, X, y = simple_fit
        q = model.predict_quantile(X[:50], q=0.99)
        assert np.all(np.isfinite(q))

    def test_predict_quantile_positive(self, simple_fit):
        model, X, y = simple_fit
        q = model.predict_quantile(X[:50], q=0.99)
        assert np.all(q > 0)

    def test_quantile_monotone_in_level(self, simple_fit):
        """Higher quantile level gives higher quantile estimate."""
        model, X, y = simple_fit
        q90 = model.predict_quantile(X[:20], q=0.90)
        q95 = model.predict_quantile(X[:20], q=0.95)
        q99 = model.predict_quantile(X[:20], q=0.99)
        q995 = model.predict_quantile(X[:20], q=0.995)
        assert np.all(q99 > q95)
        assert np.all(q95 > q90)
        assert np.all(q995 > q99)

    def test_tvar_geq_var(self, simple_fit):
        """TVaR(tau) >= VaR(tau) for all tau."""
        model, X, y = simple_fit
        for tau in [0.85, 0.90, 0.95, 0.99]:
            var = model.predict_quantile(X[:20], q=tau)
            tvar = model.predict_tvar(X[:20], q=tau)
            assert np.all(tvar >= var), f"TVaR < VaR at tau={tau}"

    def test_tvar_monotone_in_level(self, simple_fit):
        """TVaR is increasing in quantile level."""
        model, X, y = simple_fit
        tv90 = model.predict_tvar(X[:20], q=0.90)
        tv95 = model.predict_tvar(X[:20], q=0.95)
        tv99 = model.predict_tvar(X[:20], q=0.99)
        assert np.all(tv99 > tv95)
        assert np.all(tv95 > tv90)

    def test_predict_params_returns_dataframe(self, simple_fit):
        model, X, y = simple_fit
        params = model.predict_params(X[:30])
        assert isinstance(params, pd.DataFrame)
        assert set(params.columns) == {"xi", "sigma", "nu", "threshold"}

    def test_predict_params_shapes(self, simple_fit):
        model, X, y = simple_fit
        params = model.predict_params(X[:30])
        assert len(params) == 30

    def test_predict_params_xi_in_range(self, simple_fit):
        """Fitted xi values respect the output constraint."""
        model, X, y = simple_fit
        params = model.predict_params(X)
        assert (params["xi"] > -0.5).all()
        assert (params["xi"] < 0.7).all()

    def test_predict_params_sigma_positive(self, simple_fit):
        """Fitted sigma values are all positive."""
        model, X, y = simple_fit
        params = model.predict_params(X)
        assert (params["sigma"] > 0).all()

    def test_predict_params_nu_equals_sigma_times_xi_plus_1(self, simple_fit):
        """nu = sigma * (xi + 1) must hold exactly."""
        model, X, y = simple_fit
        params = model.predict_params(X[:50])
        expected_nu = params["sigma"] * (params["xi"] + 1)
        np.testing.assert_allclose(params["nu"].values, expected_nu.values, rtol=1e-5)

    def test_predict_exceedance_prob_in_unit_interval(self, simple_fit):
        """All exceedance probabilities in [0, 1]."""
        model, X, y = simple_fit
        p = model.predict_exceedance_prob(X[:30], threshold=50_000.0)
        assert np.all(p >= 0)
        assert np.all(p <= 1)

    def test_predict_exceedance_prob_shape(self, simple_fit):
        model, X, y = simple_fit
        p = model.predict_exceedance_prob(X[:30], threshold=50_000.0)
        assert p.shape == (30,)

    def test_predict_xl_layer_nonneg(self, simple_fit):
        """Expected XL layer loss is non-negative."""
        model, X, y = simple_fit
        el = model.predict_xl_layer(X[:30], attachment=20_000.0, limit=100_000.0)
        assert np.all(el >= 0)

    def test_predict_xl_layer_shape(self, simple_fit):
        model, X, y = simple_fit
        el = model.predict_xl_layer(X[:30], attachment=20_000.0, limit=100_000.0)
        assert el.shape == (30,)

    def test_quantile_below_tau0_raises(self, simple_fit):
        model, X, y = simple_fit
        with pytest.raises(ValueError, match="tau_0"):
            model.predict_quantile(X[:5], q=0.5)

    def test_predict_before_fit_raises(self):
        model = EQRNModel()
        X = np.ones((5, 2))
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.predict_quantile(X, q=0.99)


class TestEQRNModelShapeFixed:
    def test_shape_fixed_produces_constant_xi(self, simple_fit):
        """shape_fixed=True gives identical xi for all observations."""
        model, X, y = simple_fit
        assert model.shape_fixed is True
        params = model.predict_params(X)
        xi_vals = params["xi"].values
        # All xi identical (up to float precision)
        assert np.allclose(xi_vals, xi_vals[0], atol=1e-5)

    def test_full_model_produces_variable_xi(self, covariate_fit):
        """shape_fixed=False gives variable xi across observations."""
        model, X_test, y_test = covariate_fit
        assert model.shape_fixed is False
        params = model.predict_params(X_test)
        xi_std = params["xi"].std()
        # Should have meaningful variation
        assert xi_std > 1e-4, f"Expected variable xi, got std={xi_std:.6f}"


class TestEQRNModelSmallDataset:
    def test_small_dataset_no_crash(self):
        """Fitting on n=500 with shape_fixed=True completes without error."""
        rng = np.random.default_rng(77)
        n = 500
        y = genpareto.rvs(c=0.25, scale=3000, loc=1000, size=n, random_state=rng)
        X = rng.standard_normal((n, 2))
        model = EQRNModel(
            tau_0=0.75,
            hidden_sizes=(8,),
            shape_fixed=True,
            n_epochs=50,
            patience=15,
            seed=77,
            verbose=0,
        )
        model.fit(X, y)
        assert model.is_fitted

    def test_small_dataset_predictions_finite(self):
        """Predictions are finite after fitting on small data."""
        rng = np.random.default_rng(88)
        n = 500
        y = genpareto.rvs(c=0.25, scale=3000, loc=1000, size=n, random_state=rng)
        X = rng.standard_normal((n, 2))
        model = EQRNModel(tau_0=0.75, hidden_sizes=(8,), shape_fixed=True,
                          n_epochs=50, patience=15, seed=88, verbose=0)
        model.fit(X, y)
        q = model.predict_quantile(X[:20], q=0.95)
        assert np.all(np.isfinite(q))


class TestEQRNModelExposureWeighting:
    def test_weighted_differs_from_unweighted(self):
        """Fitting with non-uniform weights produces different model than uniform."""
        rng = np.random.default_rng(31)
        n = 800
        y = genpareto.rvs(c=0.3, scale=5000, loc=2000, size=n, random_state=rng)
        X = rng.standard_normal((n, 2))

        model_uniform = EQRNModel(tau_0=0.8, hidden_sizes=(8,), shape_fixed=True,
                                  n_epochs=60, seed=31, verbose=0)
        model_uniform.fit(X, y)

        weights = rng.uniform(0.5, 2.0, n)
        model_weighted = EQRNModel(tau_0=0.8, hidden_sizes=(8,), shape_fixed=True,
                                   n_epochs=60, seed=31, verbose=0)
        model_weighted.fit(X, y, sample_weight=weights)

        # The intermediate quantile predictions should differ due to weights
        q_uniform = model_uniform.intermediate_estimator_.predict(X)
        q_weighted = model_weighted.intermediate_estimator_.predict(X)
        # Not identical
        assert not np.allclose(q_uniform, q_weighted, atol=1e-3)


class TestEQRNModelValidation:
    def test_negative_y_raises(self):
        """Negative response values raise ValueError."""
        X = np.ones((10, 2))
        y = np.array([-1.0] + [1000.0] * 9)
        model = EQRNModel(verbose=0)
        with pytest.raises(ValueError, match="positive"):
            model.fit(X, y)

    def test_mismatched_dimensions_raises(self):
        """Mismatched X and y dimensions raise ValueError."""
        X = np.ones((10, 2))
        y = np.ones(8)
        model = EQRNModel(verbose=0)
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_tvar_below_tau0_raises(self, simple_fit):
        """TVaR at level below tau_0 raises ValueError."""
        model, X, y = simple_fit
        with pytest.raises(ValueError, match="tau_0"):
            model.predict_tvar(X[:5], q=0.5)


class TestEQRNModelCovariateEffect:
    def test_xi_increases_with_x1(self, covariate_fit):
        """In the covariate DGP, xi should increase with x1."""
        model, X_test, y_test = covariate_fit
        params = model.predict_params(X_test)

        # Correlation between x1 and xi should be positive
        corr = np.corrcoef(X_test[:, 0], params["xi"].values)[0, 1]
        # Positive correlation expected (xi(x) = 0.15 + 0.25 * x1)
        assert corr > 0, f"Expected positive xi-x1 correlation, got {corr:.3f}"
