"""
Tests for the intermediate quantile estimator (insurance_quantile.eqrn).

Key concerns:
- OOF estimates are genuinely out-of-fold (anti-leakage check)
- Exceedance rate is close to (1 - tau_0)
- Predict works on new data
- Error handling for bad inputs
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genpareto

from insurance_quantile.eqrn.intermediate import IntermediateQuantileEstimator


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(7)
    n = 1000
    X = rng.standard_normal((n, 3))
    y = genpareto.rvs(c=0.3, scale=5000, loc=2000, size=n, random_state=rng)
    return X, y


class TestIntermediateEstimatorFit:
    def test_fit_runs_without_error(self, simple_data):
        """Basic fit completes without error."""
        X, y = simple_data
        est = IntermediateQuantileEstimator(tau_0=0.8, seed=1)
        est.fit(X, y)
        assert est._is_fitted

    def test_oof_predictions_shape(self, simple_data):
        """OOF predictions have same length as training data."""
        X, y = simple_data
        est = IntermediateQuantileEstimator(tau_0=0.8, seed=1)
        est.fit(X, y)
        assert est.oof_predictions_.shape == (len(y),)

    def test_oof_predictions_finite(self, simple_data):
        """OOF predictions are all finite."""
        X, y = simple_data
        est = IntermediateQuantileEstimator(tau_0=0.8, seed=1)
        est.fit(X, y)
        assert np.all(np.isfinite(est.oof_predictions_))

    def test_exceedance_rate_near_target(self, simple_data):
        """Fraction of y > OOF predictions should be close to 1 - tau_0."""
        X, y = simple_data
        tau_0 = 0.8
        est = IntermediateQuantileEstimator(tau_0=tau_0, seed=1)
        est.fit(X, y)
        exceed_rate = (y > est.oof_predictions_).mean()
        # Allow +-10 percentage points
        assert abs(exceed_rate - (1 - tau_0)) < 0.10, \
            f"Exceedance rate {exceed_rate:.3f} too far from target {1-tau_0:.2f}"

    def test_oof_leakage_check(self, simple_data):
        """OOF predictions must not perfectly overfit.

        If predictions are the exact quantile of training data (leakage),
        the exceedance rate would be exactly (1 - tau_0) with variance ~= 0.
        With genuine OOF, there's variance. This is a soft check: OOF
        predictions should NOT be more accurate than the full model on test.
        """
        X, y = simple_data
        n = len(y)
        est = IntermediateQuantileEstimator(tau_0=0.8, seed=1)
        est.fit(X, y)

        # OOF residuals: y_i - q_hat_oof_i
        oof_resid = y - est.oof_predictions_
        # Full model residuals: y_i - q_hat_full_i
        full_preds = est.predict(X)
        full_resid = y - full_preds

        # Full model should predict <= OOF on training data
        # (lower pinball loss because it's fitted on all data)
        tau_0 = est.tau_0
        def pinball_loss(resid, tau):
            return np.where(resid >= 0, tau * resid, (tau - 1) * resid).mean()

        oof_pinball = pinball_loss(oof_resid, tau_0)
        full_pinball = pinball_loss(full_resid, tau_0)

        # Full model has strictly lower pinball loss on training data
        assert full_pinball <= oof_pinball, \
            "OOF estimates should be less accurate than full model on training data"


class TestIntermediateEstimatorPredict:
    def test_predict_shape(self, simple_data):
        """Prediction output has correct shape."""
        X, y = simple_data
        est = IntermediateQuantileEstimator(tau_0=0.8, seed=1)
        est.fit(X, y)

        X_test = np.random.default_rng(99).standard_normal((50, 3))
        preds = est.predict(X_test)
        assert preds.shape == (50,)

    def test_predict_before_fit_raises(self):
        """predict() before fit() raises RuntimeError."""
        est = IntermediateQuantileEstimator(tau_0=0.8)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            est.predict(np.ones((5, 3)))

    def test_predict_finite(self, simple_data):
        """Predictions are finite."""
        X, y = simple_data
        est = IntermediateQuantileEstimator(tau_0=0.8, seed=1)
        est.fit(X, y)
        preds = est.predict(X[:10])
        assert np.all(np.isfinite(preds))


class TestIntermediateEstimatorValidation:
    def test_tau0_out_of_bounds_raises(self):
        """tau_0 outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError):
            IntermediateQuantileEstimator(tau_0=0.0)
        with pytest.raises(ValueError):
            IntermediateQuantileEstimator(tau_0=1.0)
        with pytest.raises(ValueError):
            IntermediateQuantileEstimator(tau_0=1.5)

    def test_n_folds_too_small_raises(self):
        """n_folds < 2 raises ValueError."""
        with pytest.raises(ValueError):
            IntermediateQuantileEstimator(n_folds=1)

    def test_sample_weight_accepted(self, simple_data):
        """Sample weights are accepted without error."""
        X, y = simple_data
        weights = np.ones(len(y)) * 2.0
        est = IntermediateQuantileEstimator(tau_0=0.8, seed=1)
        est.fit(X, y, sample_weight=weights)
        assert est._is_fitted

    def test_different_tau0_values(self, simple_data):
        """Higher tau_0 gives higher quantile predictions."""
        X, y = simple_data
        est_low = IntermediateQuantileEstimator(tau_0=0.7, seed=1)
        est_low.fit(X, y)
        est_high = IntermediateQuantileEstimator(tau_0=0.9, seed=1)
        est_high.fit(X, y)

        # Full model predictions: tau_0=0.9 should be higher quantile
        preds_low = est_low.predict(X[:20])
        preds_high = est_high.predict(X[:20])
        assert preds_high.mean() > preds_low.mean()
