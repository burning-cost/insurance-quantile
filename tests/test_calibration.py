"""
Tests for calibration diagnostics: pinball_loss and coverage_check.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_quantile import coverage_check, pinball_loss


# ---------------------------------------------------------------------------
# pinball_loss
# ---------------------------------------------------------------------------

class TestPinballLoss:
    def test_zero_loss_perfect_prediction(self):
        """If q_alpha = y (perfect prediction at all points), loss approaches 0."""
        y = pl.Series("y", [1.0, 2.0, 3.0])
        q = pl.Series("q", [1.0, 2.0, 3.0])
        assert pinball_loss(y, q, alpha=0.5) == pytest.approx(0.0)

    def test_symmetric_at_median(self):
        """At alpha=0.5, pinball loss is 0.5 * MAE."""
        y = pl.Series("y", [0.0, 2.0])
        q = pl.Series("q", [1.0, 1.0])  # q=1, y=0: residual=-1; q=1, y=2: residual=1
        # loss = 0.5*max(0,-(-1)) + 0.5*max(0,1) = (0.5*1 + 0.5*1)/2 = 0.5
        assert pinball_loss(y, q, alpha=0.5) == pytest.approx(0.5)

    def test_asymmetric_for_high_alpha(self):
        """For high alpha, underprediction is penalised more than overprediction."""
        y_under = pl.Series("y", [2.0])  # actual > predicted => underprediction
        y_over = pl.Series("y", [0.0])   # actual < predicted => overprediction
        q = pl.Series("q", [1.0])
        alpha = 0.9
        loss_under = pinball_loss(y_under, q, alpha=alpha)  # alpha * (y - q) = 0.9 * 1
        loss_over = pinball_loss(y_over, q, alpha=alpha)    # (1-alpha) * (q - y) = 0.1 * 1
        assert loss_under == pytest.approx(0.9)
        assert loss_over == pytest.approx(0.1)
        assert loss_under > loss_over

    def test_invalid_alpha_zero(self):
        y = pl.Series("y", [1.0])
        q = pl.Series("q", [1.0])
        with pytest.raises(ValueError, match="alpha must be in"):
            pinball_loss(y, q, alpha=0.0)

    def test_invalid_alpha_one(self):
        y = pl.Series("y", [1.0])
        q = pl.Series("q", [1.0])
        with pytest.raises(ValueError, match="alpha must be in"):
            pinball_loss(y, q, alpha=1.0)

    def test_pinball_positive_always(self):
        rng = np.random.default_rng(7)
        y = pl.Series("y", rng.exponential(1, 100))
        q = pl.Series("q", rng.exponential(1, 100))
        for alpha in [0.1, 0.5, 0.9]:
            assert pinball_loss(y, q, alpha=alpha) >= 0.0

    def test_better_model_lower_pinball(self, fitted_quantile_model, exponential_data):
        """A fitted model should have lower pinball loss than a constant predictor."""
        X, y = exponential_data
        preds = fitted_quantile_model.predict(X)
        model_loss = pinball_loss(y, preds["q_0.5"], alpha=0.5)

        # Constant predictor at 1.0 (roughly the mean, not the median)
        constant_pred = pl.Series("q", [1.0] * len(y))
        naive_loss = pinball_loss(y, constant_pred, alpha=0.5)
        assert model_loss < naive_loss

    def test_oracle_quantile_loss(self):
        """
        For Exponential(1), the true Q(0.9) = -ln(0.1) ≈ 2.303.
        A perfect predictor should have near-zero pinball loss on a large sample.
        """
        rng = np.random.default_rng(999)
        n = 10000
        y_np = rng.exponential(1.0, n)
        true_q90 = -np.log(0.1)  # ≈ 2.303
        y = pl.Series("y", y_np)
        q = pl.Series("q", np.full(n, true_q90))
        loss = pinball_loss(y, q, alpha=0.9)
        # The theoretical minimum pinball loss at the oracle quantile
        # For Exp(1) at alpha=0.9: approx 0.9 * E[Y - Q | Y > Q] * P(Y > Q)
        # It won't be zero, but should be small
        assert loss < 0.5


# ---------------------------------------------------------------------------
# coverage_check
# ---------------------------------------------------------------------------

class TestCoverageCheck:
    def test_returns_polars_dataframe(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        preds = fitted_quantile_model.predict(X)
        result = coverage_check(y, preds, fitted_quantile_model.spec.quantiles)
        assert isinstance(result, pl.DataFrame)

    def test_coverage_check_columns(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        preds = fitted_quantile_model.predict(X)
        result = coverage_check(y, preds, fitted_quantile_model.spec.quantiles)
        assert "quantile" in result.columns
        assert "observed_coverage" in result.columns
        assert "expected_coverage" in result.columns
        assert "coverage_error" in result.columns

    def test_coverage_values_in_range(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        preds = fitted_quantile_model.predict(X)
        result = coverage_check(y, preds, fitted_quantile_model.spec.quantiles)
        obs = result["observed_coverage"].to_list()
        for v in obs:
            assert 0.0 <= v <= 1.0

    def test_coverage_error_computation(self, fitted_quantile_model, exponential_data):
        X, y = exponential_data
        preds = fitted_quantile_model.predict(X)
        result = coverage_check(y, preds, fitted_quantile_model.spec.quantiles)
        errors = result["coverage_error"].to_numpy()
        expected = result["expected_coverage"].to_numpy()
        observed = result["observed_coverage"].to_numpy()
        np.testing.assert_allclose(errors, observed - expected, atol=1e-10)

    def test_calibrated_model_small_error(self, fitted_quantile_model, exponential_data):
        """A well-trained model on its own training data should have small coverage error."""
        X, y = exponential_data
        preds = fitted_quantile_model.predict(X)
        result = coverage_check(y, preds, fitted_quantile_model.spec.quantiles)
        max_abs_error = result["coverage_error"].abs().max()
        assert max_abs_error < 0.15, f"Large coverage error: {max_abs_error}"

    def test_coverage_increasing_with_quantile(self, fitted_quantile_model, exponential_data):
        """Observed coverage should be weakly increasing with quantile level."""
        X, y = exponential_data
        preds = fitted_quantile_model.predict(X)
        result = coverage_check(y, preds, fitted_quantile_model.spec.quantiles)
        obs = result["observed_coverage"].to_list()
        for i in range(len(obs) - 1):
            assert obs[i] <= obs[i + 1] + 0.05, "Coverage not monotone"

    def test_missing_column_raises(self, exponential_data):
        X, y = exponential_data
        bad_preds = pl.DataFrame({"q_0.5": [1.0] * len(y)})
        with pytest.raises(ValueError, match="not found in predictions"):
            coverage_check(y, bad_preds, [0.1, 0.5])

    def test_oracle_coverage(self):
        """
        Predictions exactly equal to the analytical quantiles should give
        coverage close to the nominal level.
        """
        rng = np.random.default_rng(42)
        n = 5000
        y_np = rng.exponential(1.0, n)
        y = pl.Series("y", y_np)
        qs = [0.25, 0.5, 0.75, 0.9]
        q_vals = {q: -np.log(1 - q) for q in qs}  # Exp(1) quantile formula
        preds = pl.DataFrame({f"q_{q}": np.full(n, q_vals[q]) for q in qs})
        result = coverage_check(y, preds, qs)
        for row in result.iter_rows(named=True):
            assert abs(row["coverage_error"]) < 0.05, (
                f"Oracle quantile {row['quantile']} has coverage error {row['coverage_error']}"
            )
