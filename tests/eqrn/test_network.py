"""
Tests for the GPDNet neural network module (insurance_quantile.eqrn).

Focuses on:
- Output shape correctness
- Activation constraint enforcement (xi in (-0.5, 0.7), nu > 0)
- shape_fixed behaviour
- Weight initialisation
- Parameter counting
- Batch norm and dropout variants
"""

from __future__ import annotations

import numpy as np
import pytest
torch = pytest.importorskip(
    "torch",
    reason="EQRN tests require torch. Install with: pip install insurance-quantile[eqrn]",
)

from insurance_quantile.eqrn.network import GPDNet


class TestGPDNetOutputShape:
    def test_output_shape_standard(self):
        """Forward pass returns (nu, xi) both of shape (batch_size,)."""
        net = GPDNet(input_dim=5, hidden_sizes=(8, 4))
        x = torch.randn(32, 5)
        nu, xi = net(x)
        assert nu.shape == (32,)
        assert xi.shape == (32,)

    def test_output_shape_single_observation(self):
        """Works with a single observation."""
        net = GPDNet(input_dim=3, hidden_sizes=(4,))
        x = torch.randn(1, 3)
        nu, xi = net(x)
        assert nu.shape == (1,)
        assert xi.shape == (1,)

    def test_output_shape_shape_fixed(self):
        """With shape_fixed=True, xi still broadcasts to (batch_size,)."""
        net = GPDNet(input_dim=5, hidden_sizes=(8, 4), shape_fixed=True)
        x = torch.randn(16, 5)
        nu, xi = net(x)
        assert nu.shape == (16,)
        assert xi.shape == (16,)

    def test_output_shape_large_batch(self):
        """Works with large batch."""
        net = GPDNet(input_dim=10, hidden_sizes=(16, 8, 4))
        x = torch.randn(1000, 10)
        nu, xi = net(x)
        assert nu.shape == (1000,)
        assert xi.shape == (1000,)


class TestGPDNetConstraints:
    def test_nu_positive(self):
        """All nu values must be strictly positive."""
        net = GPDNet(input_dim=8, hidden_sizes=(16, 8))
        x = torch.randn(200, 8)
        nu, xi = net(x)
        assert torch.all(nu > 0), f"Found nu <= 0: min={nu.min().item():.4f}"

    def test_xi_in_valid_range(self):
        """All xi values must be in (-0.5, 0.7)."""
        net = GPDNet(input_dim=8, hidden_sizes=(16, 8))
        x = torch.randn(200, 8)
        nu, xi = net(x)
        assert torch.all(xi > -0.5), f"Found xi <= -0.5: min={xi.min().item():.4f}"
        assert torch.all(xi < 0.7), f"Found xi >= 0.7: max={xi.max().item():.4f}"

    def test_xi_range_with_extreme_inputs(self):
        """Constraints hold even with very large input values."""
        net = GPDNet(input_dim=5, hidden_sizes=(8, 4))
        x = torch.randn(100, 5) * 100  # extreme inputs
        nu, xi = net(x)
        assert torch.all(nu > 0)
        assert torch.all(xi > -0.5)
        assert torch.all(xi < 0.7)

    def test_nu_softplus_strictly_positive(self):
        """softplus(z) > 0 for all real z; verify no exact zeros."""
        net = GPDNet(input_dim=3, hidden_sizes=(4,))
        x = torch.full((50, 3), -100.0)  # very negative inputs
        nu, xi = net(x)
        assert torch.all(nu > 0)

    def test_shape_fixed_xi_scalar(self):
        """With shape_fixed=True, all xi values are identical (one scalar)."""
        net = GPDNet(input_dim=5, hidden_sizes=(8,), shape_fixed=True)
        x = torch.randn(20, 5)
        nu, xi = net(x)
        # All xi should be the same value
        assert torch.allclose(xi, xi[0].expand_as(xi))

    def test_shape_fixed_xi_varies_with_parameter(self):
        """The scalar xi parameter changes after gradient update."""
        net = GPDNet(input_dim=3, hidden_sizes=(4,), shape_fixed=True)
        xi_before = net._xi_raw.data.clone()

        # One gradient step
        x = torch.randn(10, 3)
        z = torch.rand(10) + 0.1
        nu, xi = net(x)
        from insurance_quantile.eqrn.gpd import ogpd_loss_tensor
        loss = ogpd_loss_tensor(z, nu, xi)
        loss.backward()
        net._xi_raw.data -= 0.01 * net._xi_raw.grad

        xi_after = net._xi_raw.data
        assert not torch.allclose(xi_before, xi_after)


class TestGPDNetActivations:
    @pytest.mark.parametrize("activation", ["sigmoid", "relu", "tanh"])
    def test_all_activations_produce_valid_output(self, activation: str):
        """All supported activations produce valid (nu, xi) outputs."""
        net = GPDNet(input_dim=5, hidden_sizes=(8, 4), activation=activation)
        x = torch.randn(30, 5)
        nu, xi = net(x)
        assert torch.all(nu > 0)
        assert torch.all(xi > -0.5)
        assert torch.all(xi < 0.7)

    def test_invalid_activation_raises(self):
        """Unknown activation name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            GPDNet(input_dim=5, hidden_sizes=(4,), activation="swish")


class TestGPDNetVariants:
    def test_dropout_variant(self):
        """Network with dropout produces valid outputs during eval."""
        net = GPDNet(input_dim=5, hidden_sizes=(16, 8), p_drop=0.2)
        net.eval()
        x = torch.randn(50, 5)
        nu, xi = net(x)
        assert torch.all(nu > 0)

    def test_batch_norm_variant(self):
        """Network with batch norm produces valid outputs."""
        net = GPDNet(input_dim=5, hidden_sizes=(16, 8), batch_norm=True)
        x = torch.randn(64, 5)  # batch norm needs n > 1
        nu, xi = net(x)
        assert torch.all(nu > 0)

    def test_single_hidden_layer(self):
        """Network with one hidden layer works correctly."""
        net = GPDNet(input_dim=3, hidden_sizes=(8,))
        x = torch.randn(20, 3)
        nu, xi = net(x)
        assert nu.shape == (20,)

    def test_deep_network(self):
        """Four hidden layers works without errors."""
        net = GPDNet(input_dim=10, hidden_sizes=(32, 16, 8, 4))
        x = torch.randn(50, 10)
        nu, xi = net(x)
        assert torch.all(nu > 0)


class TestGPDNetParameters:
    def test_n_parameters_correct(self):
        """Parameter count matches manual calculation for simple network."""
        # input_dim=2, hidden=(3,): 2*3+3 = 9 (linear1)
        # hidden=(3,) to nu_head=1: 3*1+1 = 4
        # xi_head: 3*1+1 = 4
        # Total: 9 + 4 + 4 = 17
        net = GPDNet(input_dim=2, hidden_sizes=(3,), shape_fixed=False)
        # input->hidden: 2*3+3=9; nu_head: 3*1+1=4; xi_head: 3*1+1=4 -> 17
        assert net.n_parameters == 17

    def test_shape_fixed_fewer_params(self):
        """shape_fixed=True has fewer parameters than shape_fixed=False."""
        net_full = GPDNet(input_dim=5, hidden_sizes=(8, 4), shape_fixed=False)
        net_fixed = GPDNet(input_dim=5, hidden_sizes=(8, 4), shape_fixed=True)
        assert net_fixed.n_parameters < net_full.n_parameters

    def test_parameters_trainable(self):
        """All parameters require gradients by default."""
        net = GPDNet(input_dim=5, hidden_sizes=(8,))
        for p in net.parameters():
            assert p.requires_grad

    def test_predict_params_numpy_no_grad(self):
        """predict_params_numpy runs without gradient computation."""
        net = GPDNet(input_dim=5, hidden_sizes=(8,))
        x = torch.randn(20, 5)
        nu, xi = net.predict_params_numpy(x)
        assert isinstance(nu, torch.Tensor)
        assert isinstance(xi, torch.Tensor)
