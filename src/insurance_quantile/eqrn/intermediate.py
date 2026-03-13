"""
Intermediate quantile estimation for the EQRN two-step procedure.

Step 1 of EQRN requires estimating Q_hat_x(tau_0) at a moderate quantile level
tau_0 (default 0.8) for every training observation. The critical requirement is
that these estimates are OUT-OF-FOLD: the model that produced Q_hat for
observation i was not trained on observation i. Failure to enforce this
leads to artificially clean thresholds and inflated exceedance counts, which
corrupts the Step 2 GPD network.

This module implements the out-of-fold estimator using K-fold cross-validation
with LightGBM quantile regression as the default. The estimator also provides
a predict() method for new observations, which uses a model trained on the
full training set (since no leakage concern exists at inference time).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.model_selection import KFold


class IntermediateQuantileEstimator:
    """K-fold out-of-fold intermediate quantile estimator.

    Fits a quantile regression model (default: LightGBM) at level tau_0
    using K-fold cross-validation to produce out-of-fold predictions. The
    final model (trained on all data) is used for inference on new observations.

    Parameters
    ----------
    tau_0:
        Intermediate quantile level, e.g. 0.8. Must be in (0, 1).
    method:
        Quantile regression method. Currently supports 'lightgbm'.
    n_folds:
        Number of cross-validation folds. Default 5.
    seed:
        Random seed for reproducibility.
    lgbm_params:
        Additional parameters passed to LGBMRegressor. Merged with defaults.

    Notes
    -----
    The out-of-fold predictions are stored in self.oof_predictions_ after
    calling fit(). These are the thresholds used in Step 2 of EQRN.

    The full model (trained on all data) is stored in self.full_model_ and
    used by predict(). Using the full model for inference is correct because
    test data was not part of training regardless.
    """

    def __init__(
        self,
        tau_0: float = 0.8,
        method: str = "lightgbm",
        n_folds: int = 5,
        seed: Optional[int] = None,
        lgbm_params: Optional[dict] = None,
    ) -> None:
        if not 0.0 < tau_0 < 1.0:
            raise ValueError(f"tau_0 must be in (0, 1), got {tau_0}")
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")

        self.tau_0 = tau_0
        self.method = method
        self.n_folds = n_folds
        self.seed = seed
        self.lgbm_params = lgbm_params or {}

        self.oof_predictions_: Optional[np.ndarray] = None
        self.full_model_ = None
        self._is_fitted = False

    def _make_lgbm(self) -> object:
        """Instantiate a LightGBM quantile regressor."""
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError(
                "LightGBM is required for intermediate quantile estimation. "
                "Install it with: pip install lightgbm"
            )

        defaults = {
            "objective": "quantile",
            "alpha": self.tau_0,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "random_state": self.seed,
            "verbose": -1,
        }
        params = {**defaults, **self.lgbm_params}
        return LGBMRegressor(**params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "IntermediateQuantileEstimator":
        """Fit the intermediate quantile model.

        Produces out-of-fold predictions for training data and fits a full
        model for inference on new data.

        Parameters
        ----------
        X:
            Feature matrix, shape (n_samples, n_features).
        y:
            Response vector, shape (n_samples,). Should be claim severity
            values (positive).
        sample_weight:
            Optional exposure weights, shape (n_samples,).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        if X.shape[0] != n:
            raise ValueError(f"X has {X.shape[0]} rows but y has {n} elements.")

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)

        oof_preds = np.empty(n)
        oof_preds[:] = np.nan

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            model = self._make_lgbm()
            sw_train = sample_weight[train_idx] if sample_weight is not None else None
            model.fit(X[train_idx], y[train_idx], sample_weight=sw_train)
            oof_preds[val_idx] = model.predict(X[val_idx])

        # Ensure OOF predictions are within reasonable bounds
        if np.any(np.isnan(oof_preds)):
            n_nan = np.isnan(oof_preds).sum()
            raise RuntimeError(
                f"{n_nan} OOF predictions are NaN. Check input data for issues."
            )

        self.oof_predictions_ = oof_preds

        # Fit the full model for inference
        full_model = self._make_lgbm()
        full_model.fit(X, y, sample_weight=sample_weight)
        self.full_model_ = full_model
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict intermediate quantile Q_hat_x(tau_0) for new observations.

        Uses the full model trained on all training data.

        Parameters
        ----------
        X:
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted quantiles, shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        return self.full_model_.predict(X)

    @property
    def exceedance_rate(self) -> float:
        """Fraction of training observations above their predicted threshold.

        Should be close to (1 - tau_0). A substantial deviation suggests the
        intermediate quantile model is under- or over-fitting.
        """
        if self.oof_predictions_ is None:
            raise RuntimeError("Call fit() first.")
        # Not stored — computed at fit time if y is available
        raise NotImplementedError(
            "exceedance_rate requires access to y. Use model.exceedance_rate_ "
            "on the fitted EQRNModel instead."
        )
