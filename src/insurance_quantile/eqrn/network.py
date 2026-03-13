"""
Neural network architecture for EQRN.

Implements GPDNet: a feedforward neural network that maps covariates X
(plus an appended intermediate quantile feature) to the orthogonal GPD
parameters (nu(x), xi(x)).

The output activations enforce the constraints required for valid GPD
parameters:
    - nu: softplus — guarantees nu > 0
    - xi: 0.6 * tanh(z) + 0.1 — constrains xi in (-0.5, 0.7)

The orthogonal parameterisation (Pasche & Engelke 2024) makes the Fisher
information diagonal, which stabilises gradient-based training.

Optionally, xi can be treated as a single scalar parameter (shape_fixed=True),
which is a useful regularised baseline when the exceedance dataset is small
or when covariate-dependent shape variation is not expected.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPDNet(nn.Module):
    """Feedforward network mapping covariates to GPD parameters (nu, xi).

    Architecture:
        Input (p + 1 or p features) -> hidden layers -> (nu head, xi head)

    The input dimension is p + 1 when the intermediate quantile is appended
    as a feature (the default, following Pasche & Engelke 2024 Table 2).

    Parameters
    ----------
    input_dim:
        Number of input features (including the appended quantile if used).
    hidden_sizes:
        Sequence of hidden layer widths. Default (32, 16, 8) is wider than
        the CRAN default (5, 3, 3) and more appropriate for insurance data
        with ~10–30 covariates.
    activation:
        Hidden layer activation. One of 'sigmoid', 'relu', 'tanh'.
        Sigmoid is the CRAN default and tends to produce smooth parameter
        surfaces. ReLU is faster to train.
    p_drop:
        Dropout probability on hidden layers. 0 (default) means no dropout.
        Values of 0.1–0.2 are useful when n_exceedances < 500.
    shape_fixed:
        If True, xi is a single learnable scalar rather than a network output.
        Use this as a regularised baseline before fitting the full model.
    batch_norm:
        If True, apply batch normalisation after each hidden layer (before
        activation). Improves training stability for deeper networks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (32, 16, 8),
        activation: str = "sigmoid",
        p_drop: float = 0.0,
        shape_fixed: bool = False,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.shape_fixed = shape_fixed

        activation_fn = self._get_activation(activation)

        # Build hidden layers
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation_fn)
            if p_drop > 0.0:
                layers.append(nn.Dropout(p=p_drop))
            prev = h

        self.hidden = nn.Sequential(*layers)

        # Output heads
        self.nu_head = nn.Linear(prev, 1)

        if shape_fixed:
            # Scalar xi: a single parameter initialised to give xi ~ 0.1
            # 0.6 * tanh(0.0) + 0.1 = 0.1
            self._xi_raw = nn.Parameter(torch.zeros(1))
        else:
            self.xi_head = nn.Linear(prev, 1)

        self._init_weights()

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        activations = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
        }
        if name not in activations:
            raise ValueError(
                f"Unknown activation '{name}'. Choose from: {list(activations.keys())}"
            )
        return activations[name]

    def _init_weights(self) -> None:
        """Initialise weights with Xavier uniform; biases to small positive values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: map features to (nu, xi).

        Parameters
        ----------
        x:
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        nu:
            Orthogonal scale parameter, shape (batch_size,). All values > 0.
        xi:
            Shape parameter, shape (batch_size,). All values in (-0.5, 0.7).
        """
        h = self.hidden(x)

        # nu: softplus guarantees nu > 0
        nu = F.softplus(self.nu_head(h)).squeeze(-1)

        # xi: 0.6 * tanh(z) + 0.1 constrains to (-0.5, 0.7)
        if self.shape_fixed:
            xi_scalar = 0.6 * torch.tanh(self._xi_raw) + 0.1
            xi = xi_scalar.expand(nu.shape)
        else:
            xi = 0.6 * torch.tanh(self.xi_head(h)).squeeze(-1) + 0.1

        return nu, xi

    def predict_params_numpy(
        self,
        x: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict (nu, xi) with gradient disabled (inference mode).

        Parameters
        ----------
        x:
            Input tensor.
        device:
            Target device for inference. If None, uses the device of x.

        Returns
        -------
        Tuple of (nu, xi) tensors on CPU.
        """
        self.eval()
        with torch.no_grad():
            nu, xi = self.forward(x)
        return nu.cpu(), xi.cpu()

    @property
    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ShapeConstraintWarning(UserWarning):
    """Warning issued when the fitted xi distribution suggests misspecification."""
    pass


def warn_xi_distribution(xi: torch.Tensor, threshold: float = 0.5) -> None:
    """Emit a warning if the mean fitted xi is unusually high.

    Parameters
    ----------
    xi:
        Tensor of fitted xi values.
    threshold:
        Mean xi above this value triggers the warning. Default 0.5.
    """
    import warnings
    mean_xi = xi.mean().item()
    if mean_xi > threshold:
        warnings.warn(
            f"Mean fitted xi = {mean_xi:.3f} exceeds {threshold}. "
            "This may indicate model misspecification or a pathologically "
            "heavy-tailed dataset. Check the threshold stability plot.",
            ShapeConstraintWarning,
            stacklevel=2,
        )
