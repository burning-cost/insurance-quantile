# insurance-quantile

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

## Install

```bash
pip install insurance-quantile
```

## Quick start

```python
import polars as pl
from insurance_quantile import QuantileGBM, per_risk_tvar, large_loss_loading

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

# Large loss loading over a Tweedie mean model
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
```

## Expectile mode

```python
# For motor bodily injury or other heavy-tailed lines
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

QuantileGBM output feeds directly into [insurance-conformal](https://github.com/burningcost/insurance-conformal) for Conformalized Quantile Regression (CQR):

```python
from insurance_quantile import QuantileGBM
from insurance_conformal import ConformalQuantileRegressor

model = QuantileGBM(quantiles=[0.05, 0.95]).fit(X_train, y_train)
preds_cal = model.predict(X_cal)

cqr = ConformalQuantileRegressor(alpha=0.1)
cqr.fit(y_cal, preds_cal["q_0.05"], preds_cal["q_0.95"])
# Guaranteed 90% coverage, distribution-free
```

## Design decisions

**Quantile crossing fix**: isotonic regression per row at predict time. CatBoost's `MultiQuantile` loss can produce crossing predictions for individual risks despite enforcing correct orderings in the loss function. The fix is O(n_rows × n_quantiles) and adds negligible overhead.

**Exposure as sample_weight**: exposure is passed to CatBoost as `sample_weight`, not as an offset. This weights each row's loss contribution, which is appropriate when the target is aggregate cost. If your target is severity (cost per claim), do not pass exposure here.

**TVaR approximation**: we estimate TVaR by taking the mean of quantile predictions at levels above alpha. Accuracy improves with the number of high quantile levels in the model — include 0.95, 0.99 at minimum for TVaR at alpha=0.9.

**ILF integration**: `E[min(Y, L)] = integral_0^L P(Y > x) dx`, integrated numerically using the trapezoidal rule over the interpolated survival function from quantile predictions. 200 integration points is sufficient for smooth severity distributions.

## Requirements

- Python 3.10+
- catboost >= 1.2
- polars >= 0.20
- scikit-learn >= 1.3
- numpy >= 1.24

## Licence

MIT
