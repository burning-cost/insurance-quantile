"""
insurance-quantile: actuarial tail risk quantile and expectile regression.

This library wraps CatBoost's native quantile/expectile loss with an
actuarial output vocabulary for UK personal lines pricing teams.

Core class: QuantileGBM — fits quantile or expectile GBM, returns Polars
DataFrames with columns named q_0.5, q_0.9, etc.

Actuarial functions:
- per_risk_tvar / portfolio_tvar: Tail Value at Risk
- large_loss_loading: additive loading over mean burning cost
- ilf: Increased Limits Factors
- exceedance_curve / oep_curve: portfolio exceedance probability curves
- coverage_check / pinball_loss: calibration diagnostics

EQRN subpackage (insurance_quantile.eqrn):
- EQRNModel: extreme quantile regression neural network (Pasche & Engelke 2024)
- EQRNDiagnostics: GPD QQ, calibration, threshold stability plots
- GPDNet: feedforward network for covariate-dependent GPD parameters
- IntermediateQuantileEstimator: K-fold OOF intermediate quantile estimation

Integration: QuantileGBM output can feed directly into insurance-conformal
for Conformalized Quantile Regression (CQR), providing distribution-free
prediction interval guarantees on top of the GBM quantile estimates.

UK personal lines context:
- Motor own damage: use quantile mode, alpha [0.5, 0.75, 0.9, 0.95, 0.99]
- Motor bodily injury: use expectile mode (heavy tail, coherence matters)
- Property: quantile mode with high alphas for large loss loading
- Zero-inflated data (majority zero claims): model severity on non-zero
  claims; frequency separately. Document this in your model sign-off.
- Reinsurance / XL layers: use EQRNModel for covariate-dependent GPD tail
"""

from ._calibration import coverage_check, pinball_loss, quantile_calibration_plot
from ._exceedance import exceedance_curve, oep_curve
from ._loading import ilf, large_loss_loading
from ._model import QuantileGBM
from ._tvar import per_risk_tvar, portfolio_tvar
from ._types import ExceedanceCurve, QuantileSpec, TailModel, TVaRResult

# EQRN re-exports (key classes only; full API via insurance_quantile.eqrn)
from .eqrn import EQRNModel, EQRNDiagnostics, GPDNet, IntermediateQuantileEstimator

__version__ = "0.2.3"

__all__ = [
    # Core GBM model
    "QuantileGBM",
    # Types
    "QuantileSpec",
    "TailModel",
    "TVaRResult",
    "ExceedanceCurve",
    # TVaR
    "per_risk_tvar",
    "portfolio_tvar",
    # Loading / ILF
    "large_loss_loading",
    "ilf",
    # Exceedance
    "exceedance_curve",
    "oep_curve",
    # Calibration
    "coverage_check",
    "pinball_loss",
    "quantile_calibration_plot",
    # EQRN (extreme quantile neural net)
    "EQRNModel",
    "EQRNDiagnostics",
    "GPDNet",
    "IntermediateQuantileEstimator",
    # Version
    "__version__",
]
