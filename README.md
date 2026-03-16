# insurance-quantile
[![Tests](https://github.com/burning-cost/insurance-quantile/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-quantile/actions/workflows/tests.yml)

Actuarial tail risk quantile and expectile regression for UK personal lines pricing.

Wraps CatBoost's native `MultiQuantile` loss with the vocabulary actuaries actually use: TVaR, large loss loading, ILFs, OEP curves. Polars in, Polars out.

## The problem

Tweedie GBMs estimate E[Y | X] well. But pricing teams routinely need:

- **Large loss loading**: how much extra to add for claims that blow past the mean
- **Increased Limits Factors**: what to charge for higher policy limits
- **TVaR per risk**: expected loss given it exceeds its VaR threshold
- **OEP curves**: exceedance probability for reinsurance attachment points

None of this comes out of a Tweedie model. You need the full conditional distribution, or at least a set of quantiles.

## Why CatBoost quantile regression, not quantile forests or GLMs

CatBoost's `MultiQuantile` loss trains a single model for all quantile levels simultaneously — shared feature representations, one training pass. It outperforms separate models and is faster to fit than quantile random forests on structured tabular data. The downside is quantile crossing at prediction time (CatBoost issue #2317), which we fix with per-row isotonic regression.

For heavy-tailed lines (motor BI, liability), expectile mode is available. Expectile regression has a property quantile regression lacks: it is both **coherent** (satisfies subadditivity) and **elicitable** (has a proper scoring rule). This makes it backtestable and suitable for ORSA and Solvency II reporting.

## Installation

```bash
pip install insurance-quantile
```

## Quick start

```python
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from insurance_quantile import QuantileGBM, per_risk_tvar, large_loss_loading

# Synthetic motor severity portfolio — 1,000 non-zero claims
rng = np.random.default_rng(42)
n = 1_000

vehicle_age   = rng.integers(1, 15, n).astype(float)
driver_age    = rng.integers(21, 75, n).astype(float)
ncd_years     = rng.integers(0, 9, n).astype(float)
vehicle_group = rng.choice([1.0, 2.0, 3.0, 4.0], size=n)  # encoded as float
exposure      = rng.uniform(0.3, 1.0, n)

# Heteroskedastic lognormal severity: tail weight increases with vehicle group
log_mu    = 6.5 + 0.03 * vehicle_age - 0.01 * ncd_years + 0.1 * vehicle_group
log_sigma = 0.5 + 0.05 * vehicle_group   # tail weight varies by segment
claim_amount = np.exp(rng.normal(log_mu, log_sigma, n))

# Feature matrix — QuantileGBM requires Polars input
feature_names = ["vehicle_age", "driver_age", "ncd_years", "vehicle_group"]
X = pl.DataFrame({
    "vehicle_age":   vehicle_age,
    "driver_age":    driver_age,
    "ncd_years":     ncd_years,
    "vehicle_group": vehicle_group,
})
y = pl.Series("claim_amount", claim_amount)

idx_train, idx_val = train_test_split(np.arange(n), test_size=0.2, random_state=42)
X_train, X_val       = X[idx_train], X[idx_val]
y_train, y_val       = y[idx_train], y[idx_val]
exposure_train       = pl.Series("exposure", exposure[idx_train])

# Fit quantile GBM
model = QuantileGBM(
    quantiles=[0.5, 0.75, 0.9, 0.95, 0.99],
    fix_crossing=True,
    iterations=500,
)
model.fit(X_train, y_train, exposure=exposure_train)

# Predict quantiles — Polars DataFrame, columns: q_0.5, q_0.75, q_0.9, q_0.95, q_0.99
preds = model.predict(X_val)

# TVaR per risk
tvar = per_risk_tvar(model, X_val, alpha=0.95)

# Large loss loading: requires a fitted mean model for comparison
from catboost import CatBoostRegressor
tweedie_model = CatBoostRegressor(loss_function="Tweedie:variance_power=1.5",
                                  iterations=200, verbose=0)
tweedie_model.fit(X_train.to_numpy(), y_train.to_numpy())
loading = large_loss_loading(tweedie_model, model, X_val, alpha=0.95)
```

## Module overview

```
insurance_quantile/
  QuantileGBM          — core class: fit/predict, quantile or expectile mode
  per_risk_tvar        — TVaR per risk at confidence level alpha
  portfolio_tvar       — aggregated portfolio TVaR
  large_loss_loading   — additive loading: TVaR minus mean model prediction
  ilf                  — Increased Limits Factor: E[min(Y,L2)] / E[min(Y,L1)]
  exceedance_curve     — P(Y > x) averaged across portfolio
  oep_curve            — occurrence exceedance probability (OEP)
  coverage_check       — calibration: observed vs expected coverage per quantile
  pinball_loss         — standard scoring rule for quantile regression

insurance_quantile.eqrn/
  EQRNModel            — extreme quantile regression neural network (Pasche & Engelke 2024)
  EQRNDiagnostics      — GPD QQ, calibration, threshold stability plots
  GPDNet               — feedforward network for covariate-dependent GPD parameters
  IntermediateQuantileEstimator — K-fold OOF intermediate quantile estimation
```

## Expectile mode

```python
# For motor bodily injury or other heavy-tailed lines
# X_train and y_train are pl.DataFrame and pl.Series as in the quick start above.
model = QuantileGBM(
    quantiles=[0.5, 0.75, 0.9, 0.95],
    use_expectile=True,  # fits separate CatBoost model per alpha
)
model.fit(X_train, y_train)
```

Expectiles are not the same as quantiles. The `e_0.9` expectile is generally different from `Q(0.9)`. Use expectile mode when you need a coherent, backtestable tail risk measure — not when you need P(Y > x) directly.

## Zero-inflated data

Most personal lines portfolios have a large mass of zero claims. There are two ways to handle this:

1. **Model the full distribution** (including zeros): correct but quantiles below the zero-fraction level will all be zero, which is less useful for large loss loading.
2. **Separate frequency and severity** (recommended): fit QuantileGBM only on non-zero claims (severity), with a separate frequency model. Large loss loading then applies to severity only. Document which approach you've used in your model sign-off.

## Integration with insurance-conformal

QuantileGBM output feeds directly into [insurance-conformal](https://github.com/burning-cost/insurance-conformal) for Conformalized Quantile Regression (CQR):

```python
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from insurance_quantile import QuantileGBM
from insurance_conformal import ConformalQuantileRegressor

# Assumes X and y defined as pl.DataFrame and pl.Series as in the quick start above.
# Split into train / calibration sets for conformal coverage guarantee.
idx_tr, idx_cal = train_test_split(np.arange(len(X_train) + len(X_val)),
                                   test_size=0.25, random_state=0)
X_all = pl.concat([X_train, X_val])
y_all = pl.concat([y_train, y_val])
X_tr2, X_cal2 = X_all[idx_tr], X_all[idx_cal]
y_tr2, y_cal2 = y_all[idx_tr], y_all[idx_cal]

model = QuantileGBM(quantiles=[0.05, 0.95]).fit(X_tr2, y_tr2)
preds_cal = model.predict(X_cal2)

cqr = ConformalQuantileRegressor(alpha=0.1)
cqr.fit(y_cal2, preds_cal["q_0.05"], preds_cal["q_0.95"])
# Guaranteed 90% coverage, distribution-free
```

## Design decisions

**Quantile crossing fix**: isotonic regression per row at predict time. CatBoost's `MultiQuantile` loss can produce crossing predictions for individual risks despite enforcing correct orderings in the loss function. The fix is O(n_rows × n_quantiles) and adds negligible overhead.

**Exposure as sample_weight**: exposure is passed to CatBoost as `sample_weight`, not as an offset. This weights each row's loss contribution, which is appropriate when the target is aggregate cost. If your target is severity (cost per claim), do not pass exposure here.

**TVaR approximation**: we estimate TVaR by taking the mean of quantile predictions at levels above alpha. Accuracy improves with the number of high quantile levels in the model — include 0.95, 0.99 at minimum for TVaR at alpha=0.9.

**ILF integration**: `E[min(Y, L)] = integral_0^L P(Y > x) dx`, integrated numerically using the trapezoidal rule over the interpolated survival function from quantile predictions. 200 integration points is sufficient for smooth severity distributions.

---

## Performance

Benchmarked against **parametric lognormal quantiles** (OLS on log(Y) with global sigma) on synthetic severity data — 5,000 claims from a heteroskedastic lognormal DGP where tail weight (`log_sigma`) varies by vehicle group from 0.46 to 0.64. 4,000 train / 1,000 test split. Numbers from the post-P0-fix benchmark run:

**TABLE 1: Quantile calibration (coverage) and pinball loss**

| Quantile | Coverage — Lognormal | Coverage — QuantileGBM | Pinball — Lognormal | Pinball — QuantileGBM |
|----------|---------------------|----------------------|--------------------|-----------------------|
| Q90 | 0.8970 | 0.8590 | 185.8 | 197.9 |
| Q95 | 0.9470 | 0.9150 | 122.3 | 136.7 |
| Q99 | 0.9890 | 0.9560 | 38.4 | 54.8 |

**TABLE 2: TVaR accuracy vs DGP analytical truth (TVaR_90)**

| Metric | Lognormal baseline | QuantileGBM |
|--------|--------------------|-------------|
| MAE vs DGP truth | 315.1 | 477.0 |
| RMSE vs DGP truth | 370.9 | 733.9 |
| Bias (mean over/under-estimate) | −70.9 | −39.8 |

**TABLE 3: ILF accuracy vs DGP truth (base limit £5,000)**

| Limit | ILF (DGP) | ILF (Lognormal) | ILF (QuantileGBM) | Error — Lognormal | Error — QuantileGBM |
|-------|-----------|-----------------|-------------------|-------------------|---------------------|
| £10,000 | 1.0070 | 1.0042 | 1.0029 | −0.0028 | −0.0041 |
| £25,000 | 1.0074 | 1.0043 | 1.0026 | −0.0031 | −0.0049 |
| £50,000 | 1.0074 | 1.0043 | 1.0031 | −0.0031 | −0.0043 |
| £100,000 | 1.0074 | 1.0043 | 1.0017 | −0.0031 | −0.0057 |

**TABLE 4: Q95 coverage by vehicle group (heteroskedastic test)**

| Group | True log_sigma | Coverage — Lognormal | Coverage — QuantileGBM |
|-------|---------------|---------------------|----------------------|
| 1 | 0.460 | 0.9878 | 0.9143 |
| 2 | 0.520 | 0.9421 | 0.8843 |
| 3 | 0.580 | 0.9434 | 0.9283 |
| 4 | 0.640 | 0.9153 | 0.9315 |

**Honest interpretation:** At this sample size (5,000 claims, 1,000 test), the lognormal baseline outperforms QuantileGBM on pinball loss, TVaR MAE/RMSE, and ILF accuracy for all limits tested. The GBM has lower TVaR bias (−39.8 vs −70.9), which matters when you care about directional accuracy rather than absolute error. The heteroskedastic Q95 coverage test shows QuantileGBM adapts better for group 4 (heaviest tail, 0.9315 vs 0.9153) but worse for groups 1–2.

The OLS lognormal benefits from the fact that the DGP is lognormal — a correctly specified parametric model will win when the distribution family is right. The GBM advantage appears at larger sample sizes and when the true distribution diverges from lognormal: e.g. bimodal severity (small repair + large BI), threshold effects by region, or mixing across lines. Run the full Databricks notebook for larger-scale validation.

**When to use:** Where severity genuinely departs from the parametric family you'd otherwise assume, and you have more than ~10,000 non-zero claims. For TVaR and large loss loading where the tail shape varies across risk segments in a way that a single-parameter family cannot capture.

**When NOT to use:** When the portfolio has fewer than ~5,000 large claims — parametric regularisation wins at small n. When the actuarial deliverable requires a smooth, monotone ILF curve — quantile regression is not constrained to be monotone in the limit dimension without additional work.



## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_quantile_demo.py).

## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Conformalized Quantile Regression — wraps this library's output to give distribution-free coverage guarantees |
| [insurance-distributional](https://github.com/burning-cost/insurance-distributional) | Parametric severity distributions (Pareto, Gamma, LogNormal) — alternative approach when you need closed-form tail quantities |
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract what's driving the tail — SHAP values on the QuantileGBM output |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation for time-structured insurance data |

[All Burning Cost libraries →](https://burning-cost.github.io)

## Read more

[Your Burning Cost Has a Tail Risk Problem](https://burning-cost.github.io/blog/insurance-quantile) — why Tweedie models systematically misprice tail risk and how quantile regression fills the gap.

## Source repos

This package consolidates two previously separate libraries:

- `insurance-quantile` — core CatBoost quantile/expectile GBM (v0.1.x)
- `insurance-eqrn` — archived, merged into `insurance_quantile.eqrn`

## Requirements

- Python 3.10+
- catboost >= 1.2
- polars >= 1.0
- scikit-learn >= 1.3
- numpy >= 1.24


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-severity](https://github.com/burning-cost/insurance-severity) | Composite Pareto-Gamma severity models — parametric alternative when closed-form ILFs and tail quantities are required |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Conformal prediction intervals — wrap quantile GBM output with distribution-free coverage guarantees |
| [insurance-distributional](https://github.com/burning-cost/insurance-distributional) | Parametric severity distributions — alternative approach when you need the full distributional shape, not just quantile levels |

## Licence

MIT
