"""
Shared fixtures for the insurance-quantile test suite.

We use synthetic distributions where quantiles are known analytically,
so tests can verify correctness rather than just checking that the code
runs without error.

Distributions used:
    - Exponential(rate=1): Q(alpha) = -ln(1-alpha), E[Y] = 1.0, TVaR_0.9 = 1 - ln(0.1) ≈ 3.302
    - Lognormal(mu=0, sigma=1): Q(alpha) = exp(Phi^-1(alpha))
    - Pareto(alpha=2, scale=1): heavy tail for large loss loading tests

All synthetic datasets are small enough for fast CatBoost training
(n <= 5000 rows, low-dimensional features).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------

def _make_exponential_data(
    n: int = 2000,
    n_features: int = 3,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.Series]:
    """
    Generate synthetic data where the target is Exponential(rate=1).

    Features are uniform noise — there is no true signal, so the
    model will learn approximately constant quantiles matching the
    marginal distribution. Analytical quantiles: Q(alpha) = -ln(1-alpha).
    """
    rng = np.random.default_rng(seed)
    X_np = rng.uniform(-1, 1, size=(n, n_features))
    y_np = rng.exponential(scale=1.0, size=n)
    X = pl.DataFrame({f"x{i}": X_np[:, i] for i in range(n_features)})
    y = pl.Series("y", y_np)
    return X, y


def _make_lognormal_data(
    n: int = 2000,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.Series]:
    """Generate lognormal data with a single feature."""
    rng = np.random.default_rng(seed)
    X = pl.DataFrame({"x": rng.uniform(0, 1, size=n)})
    y = pl.Series("y", rng.lognormal(mean=mu, sigma=sigma, size=n))
    return X, y


def _make_heterogeneous_data(
    n: int = 3000,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.Series]:
    """
    Generate data where the conditional distribution depends on X.

    Y | X ~ Exponential(rate = exp(0.5 * x1))
    So the conditional Q(0.9 | x1) = -ln(0.1) / exp(0.5 * x1)
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, size=n)
    x2 = rng.uniform(-1, 1, size=n)
    rates = np.exp(0.5 * x1)
    y = rng.exponential(scale=1.0 / rates, size=n)
    X = pl.DataFrame({"x1": x1, "x2": x2})
    y_series = pl.Series("y", y)
    return X, y_series


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def exponential_data() -> tuple[pl.DataFrame, pl.Series]:
    """Exponential(1) dataset, 2000 rows, 3 noise features."""
    return _make_exponential_data(n=2000)


@pytest.fixture(scope="session")
def lognormal_data() -> tuple[pl.DataFrame, pl.Series]:
    """Lognormal(0,1) dataset, 2000 rows."""
    return _make_lognormal_data(n=2000)


@pytest.fixture(scope="session")
def heterogeneous_data() -> tuple[pl.DataFrame, pl.Series]:
    """Dataset where conditional distribution depends on x1."""
    return _make_heterogeneous_data(n=3000)


@pytest.fixture(scope="session")
def fitted_quantile_model(exponential_data):
    """
    Pre-fitted QuantileGBM (quantile mode) on exponential data.
    Session-scoped to avoid re-fitting per test.
    """
    from insurance_quantile import QuantileGBM

    X, y = exponential_data
    model = QuantileGBM(
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        use_expectile=False,
        fix_crossing=True,
        iterations=300,
        learning_rate=0.1,
        depth=4,
    )
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def fitted_expectile_model(exponential_data):
    """
    Pre-fitted QuantileGBM (expectile mode) on exponential data.
    Session-scoped.
    """
    from insurance_quantile import QuantileGBM

    X, y = exponential_data
    model = QuantileGBM(
        quantiles=[0.5, 0.75, 0.9, 0.95],
        use_expectile=True,
        fix_crossing=True,
        iterations=200,
        learning_rate=0.1,
        depth=3,
    )
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def fitted_heterogeneous_model(heterogeneous_data):
    """
    Pre-fitted QuantileGBM on heterogeneous data.
    """
    from insurance_quantile import QuantileGBM

    X, y = heterogeneous_data
    model = QuantileGBM(
        quantiles=[0.5, 0.75, 0.9, 0.95, 0.99],
        fix_crossing=True,
        iterations=400,
        learning_rate=0.05,
        depth=5,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Simple mean model wrapper for large_loss_loading tests
# ---------------------------------------------------------------------------

class _ConstantMeanModel:
    """Trivial model that returns a constant mean prediction (Polars Series)."""
    def __init__(self, mean_val: float, n: int):
        self._val = mean_val
        self._n = n

    def predict(self, X: pl.DataFrame) -> pl.Series:
        return pl.Series("mean", [self._val] * len(X))


class _CatBoostTweedieWrapper:
    """
    Wraps a CatBoostRegressor (Tweedie) to return a Polars Series from predict().
    """
    def __init__(self, cb_model):
        self._model = cb_model

    def predict(self, X: pl.DataFrame) -> pl.Series:
        import numpy as np
        vals = self._model.predict(X.to_numpy().astype(np.float64))
        return pl.Series("mean", vals)
