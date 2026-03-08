# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-quantile: Full Workflow Demo
# MAGIC
# MAGIC This notebook demonstrates the complete actuarial workflow for quantile/expectile
# MAGIC GBM on UK personal lines insurance data. It covers:
# MAGIC
# MAGIC 1. Synthetic motor claim severity data generation
# MAGIC 2. Fitting a QuantileGBM (quantile mode)
# MAGIC 3. Fitting a QuantileGBM (expectile mode) — for heavy-tailed lines
# MAGIC 4. Calibration diagnostics: pinball loss and coverage check
# MAGIC 5. TVaR per risk and at portfolio level
# MAGIC 6. Large loss loading vs. a Tweedie mean model
# MAGIC 7. Increased Limits Factors (ILF)
# MAGIC 8. Exceedance probability curves (OEP)

# COMMAND ----------

# MAGIC %pip install insurance-quantile>=0.1.0 --quiet

# COMMAND ----------

import numpy as np
import polars as pl

from insurance_quantile import (
    QuantileGBM,
    coverage_check,
    exceedance_curve,
    ilf,
    large_loss_loading,
    oep_curve,
    per_risk_tvar,
    pinball_loss,
    portfolio_tvar,
)

print(f"insurance-quantile loaded OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic Motor Data
# MAGIC
# MAGIC We simulate a UK motor own damage severity portfolio. The data generating process:
# MAGIC - 10,000 policies with 3 rating factors (vehicle age, driver age, area)
# MAGIC - Claim severity conditional on a claim: Lognormal, mean = exp(mu(X))
# MAGIC - About 15% of policies have a claim (the rest are zeros — simulating a
# MAGIC   frequency × severity setup, but we model severity only here)
# MAGIC
# MAGIC This is a realistic setup: severity is right-skewed, heterogeneous by
# MAGIC risk characteristics, and the tail is where large loss loading matters.

# COMMAND ----------

rng = np.random.default_rng(42)
n = 10_000

# Rating factors
vehicle_age = rng.integers(0, 15, size=n).astype(float)
driver_age = rng.integers(18, 75, size=n).astype(float)
# Area: 0=rural, 1=suburban, 2=urban
area = rng.integers(0, 3, size=n).astype(float)

# Log-linear mean: younger drivers, older vehicles, urban area = higher severity
log_mu = (
    7.5                          # base: ~£1800 average severity
    - 0.02 * driver_age          # younger drivers: higher severity
    + 0.03 * vehicle_age         # older vehicles: higher repair cost
    + 0.15 * area                # urban premium
)

# Lognormal severity with sigma=0.8 (moderately heavy tail)
sigma = 0.8
severity = rng.lognormal(mean=log_mu, sigma=sigma, size=n)

# Exposure: earned car years, varies by policy
exposure = rng.uniform(0.3, 1.0, size=n)

X = pl.DataFrame({
    "vehicle_age": vehicle_age,
    "driver_age": driver_age,
    "area": area,
})
y = pl.Series("severity", severity)
exposure_series = pl.Series("exposure", exposure)

print(f"Dataset: {n} policies")
print(f"Severity: mean={severity.mean():.0f}, median={np.median(severity):.0f}, "
      f"p95={np.quantile(severity, 0.95):.0f}, max={severity.max():.0f}")

# COMMAND ----------

# Train / validation split
n_train = 8000
X_train = X.head(n_train)
X_val = X.tail(n - n_train)
y_train = y.head(n_train)
y_val = y.tail(n - n_train)
exp_train = exposure_series.head(n_train)

print(f"Train: {n_train} rows, Validation: {n - n_train} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit QuantileGBM (Quantile Mode)
# MAGIC
# MAGIC We fit a MultiQuantile CatBoost model covering the body and tail of the
# MAGIC severity distribution. The quantile levels are chosen to give good ILF
# MAGIC and TVaR estimates: we need good coverage above 0.9 for tail work.

# COMMAND ----------

model_q = QuantileGBM(
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    use_expectile=False,
    fix_crossing=True,
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
)
model_q.fit(X_train, y_train, exposure=exp_train)
print(f"Quantile model fitted. Features: {model_q.metadata.feature_names}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit QuantileGBM (Expectile Mode)
# MAGIC
# MAGIC For bodily injury or liability lines where the tail shape drives capital
# MAGIC requirements, expectile regression is preferred. The expectile is both
# MAGIC coherent (unlike VaR) and elicitable (unlike ES/CVaR from direct estimation).
# MAGIC This makes it backtestable and suitable for ORSA / Solvency II reporting.

# COMMAND ----------

model_e = QuantileGBM(
    quantiles=[0.5, 0.75, 0.9, 0.95],
    use_expectile=True,
    fix_crossing=True,
    iterations=400,
    learning_rate=0.05,
    depth=5,
    random_seed=42,
)
model_e.fit(X_train, y_train, exposure=exp_train)
print("Expectile model fitted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Calibration Diagnostics
# MAGIC
# MAGIC We check two things on the validation set:
# MAGIC - **Coverage**: fraction of y_val <= q_alpha. Should be close to alpha.
# MAGIC - **Pinball loss**: standard scoring rule for quantile models.
# MAGIC
# MAGIC A model trained on 8k rows with 500 iterations will not be perfectly
# MAGIC calibrated, but coverage error should be small (< 0.05 per quantile).

# COMMAND ----------

preds_val = model_q.predict(X_val)
calib_report = model_q.calibration_report(X_val, y_val)

print("Coverage check (observed vs expected):")
calib_df = coverage_check(y_val, preds_val, model_q.spec.quantiles)
print(calib_df)

print(f"\nMean pinball loss: {calib_report['mean_pinball_loss']:.2f}")

# COMMAND ----------

# Individual pinball losses by quantile level
print("Pinball loss by quantile:")
for col, loss in calib_report["pinball_loss"].items():
    coverage = calib_report["coverage"][col]
    q = float(col.split("_")[1])
    print(f"  {col}: pinball={loss:.2f}, coverage={coverage:.3f} (expected {q:.2f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. TVaR Per Risk and Portfolio
# MAGIC
# MAGIC TVaR_alpha = E[Y | Y > VaR_alpha]. For alpha=0.95, this is the expected
# MAGIC severity given that the claim exceeds its 95th percentile — the figure
# MAGIC relevant for large loss XL treaty pricing and catastrophe capital.

# COMMAND ----------

tvar_result = per_risk_tvar(model_q, X_val, alpha=0.95)

print(f"Per-risk TVaR at 95% confidence:")
print(f"  Mean TVaR_0.95 = £{float(tvar_result.values.mean()):,.0f}")
print(f"  Mean VaR_0.95  = £{float(tvar_result.var_values.mean()):,.0f}")
print(f"  Mean loading (TVaR - VaR) = £{float(tvar_result.loading_over_var.mean()):,.0f}")

portfolio_mean_tvar = portfolio_tvar(model_q, X_val, alpha=0.95, aggregate_method="mean")
portfolio_sum_tvar = portfolio_tvar(model_q, X_val, alpha=0.95, aggregate_method="sum")
print(f"\nPortfolio TVaR_0.95: mean = £{portfolio_mean_tvar:,.0f}, sum = £{portfolio_sum_tvar:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Large Loss Loading
# MAGIC
# MAGIC The large loss loading is the additive adjustment over the pure burning
# MAGIC cost mean to cover tail losses. We use a simple CatBoost Tweedie model
# MAGIC as the mean reference model.

# COMMAND ----------

from catboost import CatBoostRegressor

# Fit a Tweedie mean model (standard burning cost)
tweedie_model_raw = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=300,
    learning_rate=0.05,
    depth=5,
    random_seed=42,
    verbose=0,
)
tweedie_model_raw.fit(
    X_train.to_numpy(),
    y_train.to_numpy(),
    sample_weight=exp_train.to_numpy(),
)


# Wrap to return Polars Series
class TweedieWrapper:
    def __init__(self, model):
        self._model = model

    def predict(self, X: pl.DataFrame) -> pl.Series:
        vals = self._model.predict(X.to_numpy().astype(float))
        return pl.Series("mean", vals)


tweedie_model = TweedieWrapper(tweedie_model_raw)

# Compute per-risk large loss loading
loading = large_loss_loading(tweedie_model, model_q, X_val, alpha=0.95)

print(f"Large loss loading (TVaR_0.95 - mean_Tweedie):")
print(f"  Mean loading: £{float(loading.mean()):,.0f}")
print(f"  P25 loading:  £{float(np.quantile(loading.to_numpy(), 0.25)):,.0f}")
print(f"  P75 loading:  £{float(np.quantile(loading.to_numpy(), 0.75)):,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Increased Limits Factors (ILF)
# MAGIC
# MAGIC ILF(L1, L2) = E[min(Y, L2)] / E[min(Y, L1)].
# MAGIC
# MAGIC This is the rating factor applied to the basic limit rate to price
# MAGIC coverage at a higher limit. For example, if the basic limit is £5k
# MAGIC and you want to price to £20k, multiply the basic premium by ILF(5k, 20k).

# COMMAND ----------

# Use a small subset for ILF computation (illustration)
X_ilf = X_val.head(500)

# Limits in £ (severity is in £ from our simulation)
basic_limit = 5_000.0
higher_limit = 20_000.0

ilf_values = ilf(model_q, X_ilf, basic_limit=basic_limit, higher_limit=higher_limit)

print(f"ILF({basic_limit:,.0f}, {higher_limit:,.0f}):")
print(f"  Mean ILF:   {float(ilf_values.mean()):.4f}")
print(f"  Median ILF: {float(np.median(ilf_values.to_numpy())):.4f}")
print(f"  Range:      {float(ilf_values.min()):.3f} — {float(ilf_values.max()):.3f}")

# ILF should be > 1 for higher_limit > basic_limit
assert float(ilf_values.mean()) > 1.0, "ILF should exceed 1 when limit is raised"

# Compare a wider limit spread
ilf_wide = ilf(model_q, X_ilf, basic_limit=5_000.0, higher_limit=50_000.0)
print(f"\nILF(5k, 50k) mean: {float(ilf_wide.mean()):.4f} (wider spread = higher factor)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Exceedance Probability Curves (OEP)
# MAGIC
# MAGIC The occurrence exceedance probability (OEP) curve shows P(loss > x)
# MAGIC as a function of loss threshold x. Standard output for XL reinsurance pricing.

# COMMAND ----------

X_portfolio = X_val.head(500)

# Mean exceedance curve (default)
oep_result = oep_curve(model_q, X_portfolio, n_thresholds=50)
oep_df = oep_result.as_dataframe()

print(f"OEP curve: {oep_result.n_risks} risks, {len(oep_result.thresholds)} thresholds")
print("\nSample OEP values:")
sample_rows = oep_df.filter(pl.col("exceedance_prob") > 0.01).sample(n=5, seed=1)
print(sample_rows.sort("threshold"))

# Also compute full exceedance curve DataFrame
exc_df = exceedance_curve(model_q, X_portfolio, n_thresholds=50)
print("\nExceedance curve (first 5 rows):")
print(exc_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Integration with insurance-conformal
# MAGIC
# MAGIC The QuantileGBM output can feed directly into insurance-conformal for
# MAGIC Conformalized Quantile Regression (CQR). This gives distribution-free
# MAGIC coverage guarantees on top of the GBM quantile estimates:
# MAGIC
# MAGIC ```python
# MAGIC from insurance_quantile import QuantileGBM
# MAGIC from insurance_conformal import ConformalQuantileRegressor
# MAGIC
# MAGIC model_q = QuantileGBM(quantiles=[0.05, 0.95]).fit(X_train, y_train)
# MAGIC preds_cal = model_q.predict(X_cal)  # calibration set predictions
# MAGIC
# MAGIC cqr = ConformalQuantileRegressor(alpha=0.1)
# MAGIC cqr.fit(y_cal, preds_cal["q_0.05"], preds_cal["q_0.95"])
# MAGIC intervals = cqr.predict(model_q.predict(X_test))
# MAGIC # intervals has guaranteed 90% coverage regardless of model specification
# MAGIC ```
# MAGIC
# MAGIC The key advantage: CQR corrects for any miscalibration in the GBM quantiles
# MAGIC using the calibration set residuals. No distributional assumptions.

# COMMAND ----------

print("Demo complete. All actuarial functions demonstrated successfully.")
print("Key outputs:")
print(f"  - Quantile GBM fitted: {model_q.spec.quantiles}")
print(f"  - Expectile GBM fitted: {model_e.spec.quantiles}")
print(f"  - Mean TVaR_0.95: £{portfolio_mean_tvar:,.0f}")
print(f"  - Mean large loss loading: £{float(loading.mean()):,.0f}")
print(f"  - ILF(5k, 20k): {float(ilf_values.mean()):.3f}")
print(f"  - OEP curve: {len(oep_result.thresholds)} threshold points")
