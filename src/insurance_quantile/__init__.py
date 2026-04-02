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

Mean model integration:
- MeanModelWrapper: wraps any numpy-based regressor (CatBoost, sklearn) so it
  can be passed to large_loss_loading without Polars compatibility issues.

Two-part quantile premium:
- TwoPartQuantilePremium: frequency-severity QPP decomposition at explicit
  aggregate confidence level tau. Solves the zero-inflation problem for
  UK motor OD, property and liability pricing.
- TwoPartResult: per-policy premiums, loadings, and diagnostic fields.

Wasserstein Distributionally Robust Quantile Regression:
- WassersteinRobustQR: closed-form robust QR under p-Wasserstein ambiguity
  (Zhang, Mao & Wang 2026). O(N^{-1/2}) finite-sample guarantee, dimension-free.
  Best for thin-data segments (N < 500) at high quantiles (tau >= 0.95).
- wdrqr_large_loss_loading: robust large loss loading with distribution shift
  uncertainty formalised via Wasserstein ball radius.
- wdrqr_reserve_quantile: per-risk robust reserve quantile for Solvency II
  capital allocation.

Direct Expected Shortfall Regression:
- ExpectedShortfallRegressor: direct ES regression via the i-Rock estimator
  (Li, Zhang & He 2026, arXiv:2602.18865). Estimates ES(alpha, x) = x^T beta
  directly with asymptotic inference, no two-step quantile integration required.
  Best when you need formal SE/p-values for ES coefficients (motor BI pricing,
  Solvency II SCR at 99.5th percentile, reinsurance layer pricing).

EQRN subpackage (insurance_quantile.eqrn):
- EQRNModel: extreme quantile regression neural network (Pasche & Engelke 2024)
- EQRNDiagnostics: GPD QQ, calibration, threshold stability plots
- GPDNet: feedforward network for covariate-dependent GPD parameters
- IntermediateQuantileEstimator: K-fold OOF intermediate quantile estimation

The EQRN classes are imported lazily: they are only loaded when accessed.
This means `import insurance_quantile` does NOT require torch to be installed.
Torch and lightgbm are optional — install them with:

    pip install insurance-quantile[eqrn]

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
- Thin segments (N < 500) needing formal out-of-sample guarantee: use
  WassersteinRobustQR with the Theorem 3 eps schedule.
- ES regression with inference: use ExpectedShortfallRegressor (i-Rock)
"""

from ._calibration import coverage_check, pinball_loss, quantile_calibration_plot
from ._es_regressor import ExpectedShortfallRegressor
from ._exceedance import exceedance_curve, oep_curve
from ._loading import ilf, large_loss_loading, MeanModelWrapper
from ._model import QuantileGBM
from ._robust import WassersteinRobustQR, wdrqr_large_loss_loading, wdrqr_reserve_quantile
from ._tvar import per_risk_tvar, portfolio_tvar
from ._two_part import TwoPartQuantilePremium
from ._types import ExceedanceCurve, QuantileSpec, TailModel, TwoPartResult, TVaRResult

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-quantile")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

__all__ = [
    # Core GBM model
    "QuantileGBM",
    # Types
    "QuantileSpec",
    "TailModel",
    "TVaRResult",
    "ExceedanceCurve",
    "TwoPartResult",
    # TVaR
    "per_risk_tvar",
    "portfolio_tvar",
    # Loading / ILF
    "large_loss_loading",
    "ilf",
    "MeanModelWrapper",
    # Exceedance
    "exceedance_curve",
    "oep_curve",
    # Calibration
    "coverage_check",
    "pinball_loss",
    "quantile_calibration_plot",
    # Two-part quantile premium
    "TwoPartQuantilePremium",
    # Wasserstein robust QR
    "WassersteinRobustQR",
    "wdrqr_large_loss_loading",
    "wdrqr_reserve_quantile",
    # Direct ES regression (i-Rock)
    "ExpectedShortfallRegressor",
    # EQRN (extreme quantile neural net) — lazy imports, requires torch+lightgbm
    "EQRNModel",
    "EQRNDiagnostics",
    "GPDNet",
    "IntermediateQuantileEstimator",
    # Version
    "__version__",
]


def __getattr__(name: str):
    """
    Lazy import for EQRN classes that require torch and lightgbm.

    torch (~2GB) and lightgbm are optional dependencies only needed for EQRN.
    We defer the import until the user actually accesses one of these classes,
    so that the rest of the library works without them installed.

    Install with: pip install insurance-quantile[eqrn]
    """
    _eqrn_names = {"EQRNModel", "EQRNDiagnostics", "GPDNet", "IntermediateQuantileEstimator"}
    if name in _eqrn_names:
        try:
            from . import eqrn as _eqrn_mod
        except ImportError as e:
            raise ImportError(
                f"'{name}' requires torch and lightgbm to be installed. "
                "Install them with: pip install insurance-quantile[eqrn]\n"
                f"Original error: {e}"
            ) from e
        obj = getattr(_eqrn_mod, name)
        # Cache on the module so subsequent accesses don't go through __getattr__
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'insurance_quantile' has no attribute '{name}'")
