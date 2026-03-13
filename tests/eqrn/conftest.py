"""
Shared fixtures for insurance_quantile.eqrn tests.

All synthetic data generation happens here. The fixtures produce realistic
heavy-tailed insurance severity datasets with known properties so tests can
verify correctness against analytical benchmarks.

Note: fixtures here use distinct names (eqrn_rng etc.) to avoid collisions
with the parent test suite's conftest.py fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def eqrn_rng() -> np.random.Generator:
    """Seeded random generator for reproducible EQRN test data."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def pareto_data(eqrn_rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simple Pareto(xi=0.4, sigma=10000) severity data, no covariates.

    Returns (X, y) where X is a single intercept column and y follows
    GPD(xi=0.4, sigma=10000) with loc=5000.
    """
    n = 2000
    from scipy.stats import genpareto
    y = genpareto.rvs(c=0.4, scale=10_000, loc=5_000, size=n, random_state=eqrn_rng)
    X = np.ones((n, 1))
    return X, y


@pytest.fixture(scope="session")
def covariate_data(eqrn_rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic data with covariate-dependent GPD shape.

    Two covariates: x1 in [0, 1], x2 binary.
    Shape: xi(x) = 0.2 + 0.2 * x1
    Scale: sigma(x) = 8000 * exp(0.3 * x2)
    Threshold: u = 10000

    The shape increases with x1 (heavier tail for higher x1).
    """
    n = 3000
    x1 = eqrn_rng.uniform(0, 1, n)
    x2 = eqrn_rng.binomial(1, 0.4, n).astype(float)
    X = np.column_stack([x1, x2])

    xi_true = 0.2 + 0.2 * x1
    sigma_true = 8000.0 * np.exp(0.3 * x2)

    from scipy.stats import genpareto
    y = np.array([
        genpareto.rvs(c=xi_true[i], scale=sigma_true[i], loc=10_000.0, random_state=int(eqrn_rng.integers(1e9)))
        for i in range(n)
    ])
    return X, y


@pytest.fixture(scope="session")
def small_dataset(eqrn_rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Small dataset (n=500) for testing stability with limited data."""
    from scipy.stats import genpareto
    n = 500
    X = eqrn_rng.standard_normal((n, 3))
    y = genpareto.rvs(c=0.3, scale=5000, loc=2000, size=n, random_state=eqrn_rng)
    return X, y


@pytest.fixture(scope="session")
def fitted_model_simple(pareto_data: tuple) -> object:
    """Fitted EQRNModel on simple Pareto data with shape_fixed=True."""
    from insurance_quantile.eqrn import EQRNModel
    X, y = pareto_data
    model = EQRNModel(
        tau_0=0.75,
        hidden_sizes=(16, 8),
        shape_fixed=True,
        n_epochs=100,
        patience=20,
        seed=42,
        verbose=0,
    )
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def fitted_model_full(covariate_data: tuple) -> object:
    """Fitted full EQRNModel on covariate-dependent data."""
    from insurance_quantile.eqrn import EQRNModel
    X, y = covariate_data
    n = len(y)
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = EQRNModel(
        tau_0=0.75,
        hidden_sizes=(16, 8),
        shape_fixed=False,
        n_epochs=150,
        patience=30,
        seed=42,
        verbose=0,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return model, X[split:], y[split:]
