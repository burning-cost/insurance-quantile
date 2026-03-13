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

Reference:
    Pasche, O.C. & Engelke, S. (2024). "Neural networks for extreme quantile
    regression with an application to forecasting of flood risk."
    Annals of Applied Statistics, 18(4), 2818-2839. DOI:10.1214/24-AOAS1907.
"""

from .model import EQRNModel
from .diagnostics import EQRNDiagnostics
from .intermediate import IntermediateQuantileEstimator
from .network import GPDNet
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
