"""
Tests for GPD distribution utilities (insurance_quantile.eqrn).

Covers:
- Quantile function: known analytical values, xi=0 limit
- Survival function: boundary conditions, monotonicity
- Log density: normalisability, support constraints
- NLL: agrees with scipy
- TVaR: closed-form correctness, xi < 1 guard
- Orthogonal loss: analytical values, xi near zero
- EQRN quantile inversion
- EQRN TVaR
- Exceedance probability
- XL layer computation
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import genpareto

from insurance_quantile.eqrn.gpd import (
    eqrn_exceedance_prob,
    eqrn_quantile,
    eqrn_tvar,
    eqrn_xl_layer,
    gpd_log_density,
    gpd_nll,
    gpd_quantile,
    gpd_survival,
    gpd_tvar,
    ogpd_loss_analytical,
    ogpd_loss_tensor,
    sigma_from_nu_xi_numpy,
)


# ---------------------------------------------------------------------------
# GPD quantile
# ---------------------------------------------------------------------------

class TestGPDQuantile:
    def test_exponential_limit(self):
        """At xi=0, GPD reduces to exponential: Q(p) = -sigma * log(1-p)."""
        p = np.array([0.5, 0.8, 0.9, 0.95])
        sigma = 5000.0
        q = gpd_quantile(p, xi=0.0, sigma=sigma)
        expected = -sigma * np.log1p(-p)
        np.testing.assert_allclose(q, expected, rtol=1e-8)

    def test_against_scipy_positive_xi(self):
        """Compare with scipy.stats.genpareto for xi=0.4."""
        p = np.array([0.5, 0.8, 0.9, 0.95, 0.99])
        xi, sigma, loc = 0.4, 10_000.0, 5_000.0
        our = gpd_quantile(p, xi=xi, sigma=sigma, loc=loc)
        scipy_q = genpareto.ppf(p, c=xi, scale=sigma, loc=loc)
        np.testing.assert_allclose(our, scipy_q, rtol=1e-7)

    def test_against_scipy_negative_xi(self):
        """Compare with scipy.stats.genpareto for xi=-0.2 (bounded tail)."""
        p = np.array([0.5, 0.8, 0.9])
        xi, sigma = -0.2, 3_000.0
        our = gpd_quantile(p, xi=xi, sigma=sigma)
        scipy_q = genpareto.ppf(p, c=xi, scale=sigma)
        np.testing.assert_allclose(our, scipy_q, rtol=1e-7)

    def test_quantile_monotone(self):
        """Quantile function is non-decreasing in p."""
        p = np.linspace(0.01, 0.99, 100)
        q = gpd_quantile(p, xi=0.3, sigma=1000.0)
        assert np.all(np.diff(q) >= 0)

    def test_loc_shift(self):
        """Location shift by loc: Q(p; loc) = Q(p; 0) + loc."""
        p = np.array([0.5, 0.9])
        q0 = gpd_quantile(p, xi=0.3, sigma=1000.0, loc=0.0)
        ql = gpd_quantile(p, xi=0.3, sigma=1000.0, loc=5000.0)
        np.testing.assert_allclose(ql, q0 + 5000.0, rtol=1e-10)

    def test_p_at_zero(self):
        """Q(0) = loc."""
        q = gpd_quantile(0.0, xi=0.3, sigma=1000.0, loc=5000.0)
        assert abs(q - 5000.0) < 1e-6

    def test_small_xi_near_zero_stable(self):
        """At xi=1e-9, result should be numerically close to exponential limit."""
        p = 0.9
        sigma = 1000.0
        q_exp = -sigma * np.log1p(-p)
        q_gpd = gpd_quantile(p, xi=1e-9, sigma=sigma)
        assert abs(q_gpd - q_exp) < 1.0  # within £1


# ---------------------------------------------------------------------------
# GPD survival
# ---------------------------------------------------------------------------

class TestGPDSurvival:
    def test_survival_at_loc_is_one(self):
        """S(loc) = 1 for any valid GPD."""
        s = gpd_survival(5000.0, xi=0.3, sigma=1000.0, loc=5000.0)
        assert abs(float(s) - 1.0) < 1e-10

    def test_survival_decreases(self):
        """Survival function is non-increasing."""
        y = np.linspace(1000, 20000, 100)
        s = gpd_survival(y, xi=0.3, sigma=5000.0)
        assert np.all(np.diff(s) <= 0)

    def test_survival_against_scipy(self):
        """Compare with scipy survival function."""
        y = np.array([1000, 5000, 10000])
        xi, sigma = 0.4, 3000.0
        our = gpd_survival(y, xi=xi, sigma=sigma)
        scipy_s = genpareto.sf(y, c=xi, scale=sigma)
        np.testing.assert_allclose(our, scipy_s, rtol=1e-7)

    def test_survival_clipped_to_unit_interval(self):
        """Survival values always in [0, 1]."""
        y = np.array([-100.0, 0.0, 1e6])
        s = gpd_survival(y, xi=0.3, sigma=1000.0)
        assert np.all((s >= 0) & (s <= 1))


# ---------------------------------------------------------------------------
# GPD log density
# ---------------------------------------------------------------------------

class TestGPDLogDensity:
    def test_against_scipy(self):
        """Log density matches scipy for xi=0.3."""
        y = np.array([500, 1000, 2000, 5000])
        xi, sigma = 0.3, 2000.0
        our = gpd_log_density(y, xi=xi, sigma=sigma)
        scipy_ld = genpareto.logpdf(y, c=xi, scale=sigma)
        np.testing.assert_allclose(our, scipy_ld, rtol=1e-7)

    def test_outside_support_gives_neg_inf(self):
        """Log density is -inf outside the support (negative excess for xi < 0)."""
        ld = gpd_log_density(-1.0, xi=-0.3, sigma=1000.0)
        assert ld == -np.inf

    def test_exponential_limit(self):
        """At xi=0, log density should equal -log(sigma) - y/sigma."""
        sigma = 2000.0
        y = 1000.0
        ld = gpd_log_density(y, xi=0.0, sigma=sigma)
        expected = -np.log(sigma) - y / sigma
        assert abs(ld - expected) < 1e-8


# ---------------------------------------------------------------------------
# GPD NLL
# ---------------------------------------------------------------------------

class TestGPDNLL:
    def test_nll_finite_valid_data(self):
        """NLL is finite for valid GPD data."""
        rng = np.random.default_rng(99)
        y = genpareto.rvs(c=0.3, scale=5000.0, size=200, random_state=rng)
        nll = gpd_nll(y, xi=0.3, sigma=5000.0)
        assert np.isfinite(nll)

    def test_nll_positive(self):
        """NLL is non-negative (negative log-likelihood)."""
        y = np.array([500.0, 1000.0, 2000.0])
        nll = gpd_nll(y, xi=0.3, sigma=3000.0)
        assert nll >= 0


# ---------------------------------------------------------------------------
# GPD TVaR
# ---------------------------------------------------------------------------

class TestGPDTVaR:
    def test_tvar_geq_quantile(self):
        """TVaR(tau) >= Q(tau) for all tau in [0.8, 0.999]."""
        taus = np.linspace(0.8, 0.999, 50)
        xi, sigma = 0.3, 5000.0
        q = gpd_quantile(taus, xi=xi, sigma=sigma)
        tv = gpd_tvar(taus, xi=xi, sigma=sigma)
        assert np.all(tv >= q)

    def test_tvar_raises_for_xi_geq_1(self):
        """TVaR raises ValueError when xi >= 1 (infinite mean)."""
        with pytest.raises(ValueError, match="xi >= 1"):
            gpd_tvar(0.9, xi=1.0, sigma=1000.0)

    def test_tvar_known_value_xi_0(self):
        """At xi=0 (exponential), TVaR(tau) = Q(tau) + sigma."""
        tau, sigma = 0.9, 2000.0
        q = gpd_quantile(tau, xi=0.0, sigma=sigma)
        tv = gpd_tvar(tau, xi=0.0, sigma=sigma)
        assert abs(tv - (q + sigma)) < 1.0


# ---------------------------------------------------------------------------
# Orthogonal GPD loss
# ---------------------------------------------------------------------------

class TestOGPDLoss:
    def test_known_analytical_value(self):
        """Test l_OGPD(1.0, 2.0, 0.3) matches manual computation."""
        z, nu, xi = 1.0, 2.0, 0.3
        expected = ogpd_loss_analytical(z, nu, xi)

        # Manual: (1 + 1/0.3) * log(1 + 0.3 * 1.3 * 1.0 / 2.0) + log(2.0) - log(1.3)
        inner = 1.0 + 0.3 * 1.3 * 1.0 / 2.0
        manual = (1.0 + 1.0 / 0.3) * np.log(inner) + np.log(2.0) - np.log(1.3)
        assert abs(expected - manual) < 1e-10

    def test_tensor_agrees_with_analytical(self):
        """Tensor implementation agrees with analytical computation."""
        z = torch.tensor([1.0, 2.0, 0.5])
        nu = torch.tensor([2.0, 3.0, 1.5])
        xi = torch.tensor([0.3, 0.2, 0.4])

        loss_tensor = ogpd_loss_tensor(z, nu, xi, reduction="none")

        for i in range(3):
            expected = ogpd_loss_analytical(z[i].item(), nu[i].item(), xi[i].item())
            assert abs(loss_tensor[i].item() - expected) < 1e-5

    def test_xi_near_zero_continuity(self):
        """At xi=1e-7, tensor loss should match exponential limit."""
        z = torch.tensor([1.0, 2.0, 3.0])
        nu = torch.tensor([2.0, 2.0, 2.0])
        xi_tiny = torch.tensor([1e-7, 1e-7, 1e-7])
        xi_zero = torch.tensor([0.0, 0.0, 0.0])  # triggers exp_case

        loss_tiny = ogpd_loss_tensor(z, nu, xi_tiny, reduction="none")
        loss_zero = ogpd_loss_tensor(z, nu, xi_zero, reduction="none")

        # Should be within 1e-3 of each other
        torch.testing.assert_close(loss_tiny, loss_zero, atol=1e-3, rtol=1e-3)

    def test_loss_mean_reduction(self):
        """Mean reduction = sum / n for all-feasible batch."""
        z = torch.tensor([0.5, 1.0, 2.0])
        nu = torch.tensor([1.0, 1.0, 1.0])
        xi = torch.tensor([0.3, 0.3, 0.3])

        loss_mean = ogpd_loss_tensor(z, nu, xi, reduction="mean")
        loss_sum = ogpd_loss_tensor(z, nu, xi, reduction="sum")
        assert abs(loss_mean.item() - loss_sum.item() / 3) < 1e-6

    def test_infeasible_masked_not_clipped(self):
        """Infeasible observations contribute zero to loss.

        For xi=-0.3, xi+1=0.7, the inner term is 1 + (-0.3)*0.7*z/nu.
        With z=10, nu=1: inner = 1 - 2.1 = -1.1 < 0 -> infeasible.
        With z=1, nu=2: inner = 1 + 0.3*1.3*1/2 = 1.195 > 0 -> feasible.
        """
        z = torch.tensor([1.0, 10.0])
        nu = torch.tensor([2.0, 1.0])
        # First obs: xi=0.3 -> inner = 1 + 0.3*1.3*1/2 = 1.195 > 0 (feasible)
        # Second obs: xi=-0.3 -> inner = 1 + (-0.3)*0.7*10/1 = -1.1 < 0 (infeasible)
        xi = torch.tensor([0.3, -0.3])

        loss_with_infeasible = ogpd_loss_tensor(z, nu, xi, reduction="none")
        feasible_loss = ogpd_loss_analytical(1.0, 2.0, 0.3)
        assert abs(loss_with_infeasible[0].item() - feasible_loss) < 1e-5
        assert loss_with_infeasible[1].item() == 0.0

    def test_loss_positive_for_valid_inputs(self):
        """Loss should be positive for typical insurance parameter values."""
        z = torch.tensor([500.0, 1000.0, 2000.0])
        nu = torch.tensor([3000.0, 3000.0, 3000.0])
        xi = torch.tensor([0.3, 0.3, 0.3])
        loss = ogpd_loss_tensor(z, nu, xi, reduction="mean")
        assert loss.item() > 0

    def test_loss_differentiable(self):
        """Loss is differentiable: gradients flow through."""
        z = torch.tensor([1.0, 2.0])
        nu = torch.tensor([2.0, 3.0], requires_grad=True)
        xi = torch.tensor([0.3, 0.2], requires_grad=True)
        loss = ogpd_loss_tensor(z, nu, xi)
        loss.backward()
        assert nu.grad is not None
        assert xi.grad is not None
        assert torch.isfinite(nu.grad).all()
        assert torch.isfinite(xi.grad).all()


# ---------------------------------------------------------------------------
# EQRN quantile inversion
# ---------------------------------------------------------------------------

class TestEQRNQuantile:
    def test_tau_equals_tau0_gives_threshold(self):
        """At tau = tau_0, quantile = threshold."""
        tau_0 = 0.8
        xi = np.array([0.3])
        sigma = np.array([5000.0])
        threshold = np.array([10_000.0])
        q = eqrn_quantile(tau_0 + 1e-6, tau_0, threshold, xi, sigma)
        # Should be very close to threshold
        assert abs(q[0] - 10_000.0) < 50.0  # small increment from tau_0+epsilon

    def test_quantile_increases_with_tau(self):
        """Higher quantile level gives higher quantile value."""
        tau_0 = 0.8
        taus = np.array([0.85, 0.9, 0.95, 0.99, 0.995])
        q = eqrn_quantile(taus, tau_0, threshold=10_000.0, xi=0.3, sigma=5000.0)
        assert np.all(np.diff(q) > 0)

    def test_xi_near_zero_stable(self):
        """At xi=1e-7, result matches log limit."""
        tau, tau_0 = 0.99, 0.8
        threshold, sigma = 10_000.0, 5_000.0
        q_gpd = eqrn_quantile(tau, tau_0, threshold, xi=1e-7, sigma=sigma)
        q_exp = threshold + sigma * np.log((1 - tau_0) / (1 - tau))
        assert abs(q_gpd - q_exp) < 10.0

    def test_matches_scipy_quantile(self):
        """EQRN quantile = scipy GPD quantile at same parameters."""
        tau, tau_0 = 0.99, 0.8
        xi, sigma, threshold = 0.4, 5000.0, 10_000.0
        q_eqrn = eqrn_quantile(tau, tau_0, threshold, xi, sigma)
        # scipy: GPD(xi, sigma, loc=threshold) at conditional prob = (tau-tau_0)/(1-tau_0)
        cond_tau = (tau - tau_0) / (1 - tau_0)
        q_scipy = genpareto.ppf(cond_tau, c=xi, scale=sigma, loc=threshold)
        assert abs(q_eqrn - q_scipy) < 1.0


# ---------------------------------------------------------------------------
# EQRN TVaR
# ---------------------------------------------------------------------------

class TestEQRNTVaR:
    def test_tvar_geq_quantile(self):
        """TVaR >= VaR for all test quantile levels."""
        tau_0 = 0.8
        taus = np.array([0.85, 0.9, 0.95, 0.99, 0.995])
        xi = np.full(5, 0.3)
        sigma = np.full(5, 5000.0)
        threshold = np.full(5, 10_000.0)

        q = eqrn_quantile(taus, tau_0, threshold, xi, sigma)
        tv = eqrn_tvar(taus, tau_0, threshold, xi, sigma)
        assert np.all(tv >= q)

    def test_tvar_increases_with_tau(self):
        """TVaR is increasing in tau."""
        tau_0 = 0.8
        taus = np.array([0.85, 0.9, 0.95, 0.99])
        tv = eqrn_tvar(taus, tau_0, threshold=10_000.0, xi=0.3, sigma=5000.0)
        assert np.all(np.diff(tv) > 0)


# ---------------------------------------------------------------------------
# EQRN exceedance probability
# ---------------------------------------------------------------------------

class TestEQRNExceedanceProb:
    def test_at_threshold_equals_1_minus_tau0(self):
        """P(Y > threshold | X=x) = 1 - tau_0 at the threshold."""
        tau_0 = 0.8
        threshold = np.array([10_000.0])
        xi = np.array([0.3])
        sigma = np.array([5000.0])
        p = eqrn_exceedance_prob(threshold, tau_0, threshold, xi, sigma)
        assert abs(p[0] - (1 - tau_0)) < 1e-6

    def test_decreases_with_value(self):
        """P(Y > y | X) is non-increasing in y."""
        tau_0 = 0.8
        y_vals = np.array([10_000.0, 20_000.0, 50_000.0, 100_000.0])
        threshold = np.full(4, 10_000.0)
        xi = np.full(4, 0.3)
        sigma = np.full(4, 5000.0)
        p = eqrn_exceedance_prob(y_vals, tau_0, threshold, xi, sigma)
        assert np.all(np.diff(p) <= 0)

    def test_probability_in_unit_interval(self):
        """All probabilities are in [0, 1]."""
        tau_0 = 0.8
        y = np.array([5_000.0, 10_000.0, 50_000.0, 1_000_000.0])
        threshold = np.full(4, 10_000.0)
        xi = np.full(4, 0.3)
        sigma = np.full(4, 5000.0)
        p = eqrn_exceedance_prob(y, tau_0, threshold, xi, sigma)
        assert np.all((p >= 0) & (p <= 1))


# ---------------------------------------------------------------------------
# XL layer
# ---------------------------------------------------------------------------

class TestXLLayer:
    def test_nonnegative(self):
        """Expected layer loss is always >= 0."""
        tau_0 = 0.8
        xi = np.array([0.3, 0.1, 0.4])
        sigma = np.array([5000.0, 3000.0, 8000.0])
        threshold = np.array([10_000.0, 8_000.0, 12_000.0])
        el = eqrn_xl_layer(15_000.0, 500_000.0, tau_0, threshold, xi, sigma)
        assert np.all(el >= 0)

    def test_increases_with_limit(self):
        """Expected loss increases with layer limit."""
        tau_0 = 0.8
        xi = np.array([0.3])
        sigma = np.array([5000.0])
        threshold = np.array([10_000.0])
        el_small = eqrn_xl_layer(15_000.0, 10_000.0, tau_0, threshold, xi, sigma)
        el_large = eqrn_xl_layer(15_000.0, 100_000.0, tau_0, threshold, xi, sigma)
        assert el_large[0] > el_small[0]

    def test_zero_limit_gives_zero(self):
        """Zero layer limit gives zero expected loss."""
        tau_0 = 0.8
        xi = np.array([0.3])
        sigma = np.array([5000.0])
        threshold = np.array([10_000.0])
        el = eqrn_xl_layer(15_000.0, 0.0, tau_0, threshold, xi, sigma)
        assert abs(el[0]) < 1e-6


# ---------------------------------------------------------------------------
# Sigma recovery
# ---------------------------------------------------------------------------

class TestSigmaFromNuXi:
    def test_recovery(self):
        """sigma = nu / (xi + 1)."""
        nu = np.array([1.3, 2.5, 0.8])
        xi = np.array([0.3, 0.2, 0.4])
        sigma = sigma_from_nu_xi_numpy(nu, xi)
        expected = nu / (xi + 1.0)
        np.testing.assert_allclose(sigma, expected, rtol=1e-10)

    def test_sigma_positive_for_valid_xi(self):
        """sigma > 0 when nu > 0 and xi > -1."""
        nu = np.array([0.5, 1.0, 2.0])
        xi = np.array([-0.3, 0.0, 0.5])
        sigma = sigma_from_nu_xi_numpy(nu, xi)
        assert np.all(sigma > 0)
