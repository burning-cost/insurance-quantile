"""
insurance_quantile.eqrn: Extreme Quantile Regression Neural Networks.

Implements the EQRN method of Pasche & Engelke (2024, Annals of Applied Statistics):
covariate-dependent GPD tail modelling via neural network. The first Python
implementation of conditional-EVT with neural network flexibility.

Absorbed from insurance-eqrn (v0.1.1) into insurance-quantile v0.2.0.

Primary interface::

    from insurance_quantile.eqrn import EQRNModel, EQRNDiagnostics

    model = EQRNModel(tau_0=0.8, hidden_sizes=(32, 16, 8))
    model.fit(X_train, y_train)

    q995 = model.predict_quantile(X_test, q=0.995)
    tvar_99 = model.predict_tvar(X_test, q=0.99)

Torch and lightgbm are optional dependencies — install with:

    pip install insurance-quantile[eqrn]

Reference:
    Pasche, O.C. & Engelke, S. (2024). "Neural networks for extreme quantile
    regression with an application to forecasting of flood risk."
    Annals of Applied Statistics, 18(4), 2818-2839. DOI:10.1214/24-AOAS1907.
"""

# Exports from pure-numpy submodules (no torch required)
from .gpd import (
    gpd_quantile,
    gpd_survival,
    gpd_log_density,
    gpd_nll,
    gpd_tvar,
    eqrn_quantile,
    eqrn_tvar,
    eqrn_exceedance_prob,
    eqrn_xl_layer,
    ogpd_loss_tensor,
    ogpd_loss_analytical,
)

__all__ = [
    "EQRNModel",
    "EQRNDiagnostics",
    "IntermediateQuantileEstimator",
    "GPDNet",
    "gpd_quantile",
    "gpd_survival",
    "gpd_log_density",
    "gpd_nll",
    "gpd_tvar",
    "eqrn_quantile",
    "eqrn_tvar",
    "eqrn_exceedance_prob",
    "eqrn_xl_layer",
    "ogpd_loss_tensor",
    "ogpd_loss_analytical",
]


def __getattr__(name: str):
    """
    Lazy import for classes that require torch: EQRNModel, EQRNDiagnostics,
    GPDNet, IntermediateQuantileEstimator.

    Importing this subpackage does not require torch. Torch is only loaded
    when you access one of these classes. Install with:

        pip install insurance-quantile[eqrn]
    """
    _torch_classes = {"EQRNModel", "EQRNDiagnostics", "GPDNet", "IntermediateQuantileEstimator"}
    if name in _torch_classes:
        try:
            import torch as _  # noqa: F401 — validate torch is available
        except ImportError:
            raise ImportError(
                f"'{name}' requires torch and lightgbm to be installed. "
                "Install them with: pip install insurance-quantile[eqrn]"
            )
        if name == "EQRNModel":
            from .model import EQRNModel
            globals()["EQRNModel"] = EQRNModel
            return EQRNModel
        if name == "EQRNDiagnostics":
            from .diagnostics import EQRNDiagnostics
            globals()["EQRNDiagnostics"] = EQRNDiagnostics
            return EQRNDiagnostics
        if name == "GPDNet":
            from .network import GPDNet
            globals()["GPDNet"] = GPDNet
            return GPDNet
        if name == "IntermediateQuantileEstimator":
            from .intermediate import IntermediateQuantileEstimator
            globals()["IntermediateQuantileEstimator"] = IntermediateQuantileEstimator
            return IntermediateQuantileEstimator
    raise AttributeError(f"module 'insurance_quantile.eqrn' has no attribute '{name}'")
