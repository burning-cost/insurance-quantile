"""
EQRNModel: the main user-facing class for EQRN.

Implements the two-step Extreme Quantile Regression Neural Network of
Pasche & Engelke (2024). The API follows scikit-learn conventions where
possible: fit(), predict_*() methods, stored fitted attributes with trailing
underscores.

Typical usage for UK motor TPBI pricing::

    from insurance_eqrn import EQRNModel

    model = EQRNModel(tau_0=0.85, hidden_sizes=(32, 16, 8), n_epochs=300)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # 99.5th VaR per risk profile
    var_995 = model.predict_quantile(X_test, q=0.995)

    # TVaR for reinsurance layer pricing
    tvar_99 = model.predict_tvar(X_test, q=0.99)

    # XL layer expected loss: £500k xs £500k
    xl = model.predict_xl_layer(X_test, attachment=500_000, limit=500_000)
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .gpd import (
    eqrn_exceedance_prob,
    eqrn_quantile,
    eqrn_tvar,
    eqrn_xl_layer,
    ogpd_loss_tensor,
    sigma_from_nu_xi_numpy,
)
from .intermediate import IntermediateQuantileEstimator
from .network import GPDNet, warn_xi_distribution


class EQRNModel:
    """Extreme Quantile Regression Neural Network model.

    Two-step procedure:
        1. Estimate conditional intermediate quantile Q_hat_x(tau_0) via
           K-fold quantile regression (LightGBM by default). Out-of-fold
           predictions are mandatory to prevent leakage into Step 2.
        2. Train a feedforward neural network mapping (X, Q_hat_x(tau_0))
           to GPD parameters (nu(x), xi(x)) using the orthogonal GPD
           deviance loss on observations exceeding the intermediate quantile.

    Parameters
    ----------
    tau_0:
        Intermediate quantile level. Default 0.8. At this level ~20% of
        observations contribute to GPD training. Increase (e.g. 0.85, 0.9)
        for smaller datasets to reduce noise; decrease (e.g. 0.7) for very
        large datasets where low-tail observations improve estimates.
    hidden_sizes:
        Hidden layer widths for the GPD network. Default (32, 16, 8) is
        appropriate for 10–30 insurance covariates. Use narrower networks
        (e.g. (16, 8)) for small exceedance datasets.
    activation:
        Hidden layer activation: 'sigmoid' (default), 'relu', or 'tanh'.
    p_drop:
        Dropout probability. Default 0. Try 0.1–0.2 if n_exceedances < 500.
    shape_fixed:
        If True, xi is a single scalar rather than a network output. This
        is the regularised baseline — fit this first before the full model.
    n_epochs:
        Maximum training epochs. Default 500.
    lr:
        Adam learning rate. Default 1e-4.
    batch_size:
        Mini-batch size. Automatically reduced if n_exceedances < batch_size.
    patience:
        Early stopping patience (epochs without improvement on validation
        loss). Default 50. Uses training loss if no validation set provided.
    l2_pen:
        L2 weight decay (AdamW). Default 1e-4.
    shape_penalty:
        Additional L2 penalty on the variance of xi(x) predictions across
        each batch. Encourages smoother xi surfaces. Default 0.
    intermediate_method:
        Method for intermediate quantile. Currently only 'lightgbm'.
    n_folds:
        K for KFold OOF intermediate quantile estimation.
    intermediate_model:
        Pre-fitted IntermediateQuantileEstimator. If provided, Step 1 is
        skipped and this model is used directly.
    append_quantile_feature:
        If True (default), appends Q_hat_x(tau_0) as an input feature to
        the GPD network. Consistently improves performance per Pasche &
        Engelke (2024) Table 2.
    scale_features:
        If True (default), standardises input features before network training.
        Stores mean/std for consistent scaling at inference time.
    batch_norm:
        If True, applies batch normalisation in the network.
    seed:
        Random seed for reproducibility.
    device:
        PyTorch device. If None, uses CUDA if available, else CPU.
    verbose:
        Verbosity level. 0 = silent, 1 = epoch summaries, 2 = all.
    """

    def __init__(
        self,
        tau_0: float = 0.8,
        hidden_sizes: Sequence[int] = (32, 16, 8),
        activation: str = "sigmoid",
        p_drop: float = 0.0,
        shape_fixed: bool = False,
        n_epochs: int = 500,
        lr: float = 1e-4,
        batch_size: int = 256,
        patience: int = 50,
        l2_pen: float = 1e-4,
        shape_penalty: float = 0.0,
        intermediate_method: str = "lightgbm",
        n_folds: int = 5,
        intermediate_model: Optional[IntermediateQuantileEstimator] = None,
        append_quantile_feature: bool = True,
        scale_features: bool = True,
        batch_norm: bool = False,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: int = 1,
    ) -> None:
        self.tau_0 = tau_0
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        self.p_drop = p_drop
        self.shape_fixed = shape_fixed
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.l2_pen = l2_pen
        self.shape_penalty = shape_penalty
        self.intermediate_method = intermediate_method
        self.n_folds = n_folds
        self.intermediate_model = intermediate_model
        self.append_quantile_feature = append_quantile_feature
        self.scale_features = scale_features
        self.batch_norm = batch_norm
        self.seed = seed
        self.verbose = verbose

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Fitted attributes (set during fit())
        self.network_: Optional[GPDNet] = None
        self.intermediate_estimator_: Optional[IntermediateQuantileEstimator] = None
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None
        self.train_losses_: list[float] = []
        self.val_losses_: list[float] = []
        self.n_exceedances_: Optional[int] = None
        self.exceedance_rate_: Optional[float] = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "EQRNModel":
        """Fit the two-step EQRN model.

        Parameters
        ----------
        X:
            Training covariate matrix, shape (n_train, n_features).
        y:
            Training response (claim severity), shape (n_train,). All values
            must be positive.
        X_val:
            Optional validation covariate matrix. Used for early stopping.
        y_val:
            Optional validation response. Required if X_val is provided.
        sample_weight:
            Optional sample weights for the intermediate quantile Step 1.
            The GPD Step 2 network does not currently support observation-level
            weights (uniform weighting over exceedances).

        Returns
        -------
        self
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        if np.any(y <= 0):
            raise ValueError(
                "All response values must be positive (claim severities). "
                f"Found {(y <= 0).sum()} non-positive values."
            )

        # ----------------------------------------------------------------
        # Step 1: Intermediate quantile estimation (out-of-fold)
        # ----------------------------------------------------------------
        if self.intermediate_model is not None:
            # Use provided pre-fitted estimator
            self.intermediate_estimator_ = self.intermediate_model
            q_train = self.intermediate_estimator_.predict(X)
            # We still need OOF predictions; recompute from the estimator
            if hasattr(self.intermediate_estimator_, "oof_predictions_") and \
               self.intermediate_estimator_.oof_predictions_ is not None:
                q_train_oof = self.intermediate_estimator_.oof_predictions_
            else:
                warnings.warn(
                    "Pre-fitted intermediate_model has no oof_predictions_. "
                    "Using predict(X_train) instead — this may cause leakage.",
                    UserWarning,
                    stacklevel=2,
                )
                q_train_oof = q_train
        else:
            if self.verbose >= 1:
                print(
                    f"[EQRN] Step 1: fitting {self.intermediate_method} quantile "
                    f"regression at tau_0={self.tau_0} with {self.n_folds}-fold CV..."
                )
            estimator = IntermediateQuantileEstimator(
                tau_0=self.tau_0,
                method=self.intermediate_method,
                n_folds=self.n_folds,
                seed=self.seed,
            )
            estimator.fit(X, y, sample_weight=sample_weight)
            self.intermediate_estimator_ = estimator
            q_train_oof = estimator.oof_predictions_

        # Exceedance set: observations above their OOF predicted threshold
        exceed_mask = y > q_train_oof
        n_exceed = exceed_mask.sum()
        self.n_exceedances_ = int(n_exceed)
        self.exceedance_rate_ = float(n_exceed / n)

        if self.verbose >= 1:
            print(
                f"[EQRN] Exceedances: {n_exceed}/{n} "
                f"({100 * self.exceedance_rate_:.1f}%, expected ~{100*(1-self.tau_0):.0f}%)"
            )

        if n_exceed < 50:
            warnings.warn(
                f"Only {n_exceed} exceedances above the intermediate threshold. "
                "EQRN estimates will be unstable. Consider lowering tau_0 or "
                "using shape_fixed=True.",
                UserWarning,
                stacklevel=2,
            )

        # ----------------------------------------------------------------
        # Build network inputs for exceedances
        # ----------------------------------------------------------------
        X_exc = X[exceed_mask]
        z_exc = y[exceed_mask] - q_train_oof[exceed_mask]  # excess responses
        q_exc = q_train_oof[exceed_mask]  # thresholds for exceedances

        # Assemble features: [X | Q_hat] if append_quantile_feature
        X_net_exc = self._build_network_input(X_exc, q_exc)

        # ----------------------------------------------------------------
        # Feature standardisation
        # ----------------------------------------------------------------
        if self.scale_features:
            self.feature_mean_ = X_net_exc.mean(axis=0)
            self.feature_std_ = X_net_exc.std(axis=0) + 1e-8
            X_net_exc = (X_net_exc - self.feature_mean_) / self.feature_std_
        else:
            self.feature_mean_ = np.zeros(X_net_exc.shape[1])
            self.feature_std_ = np.ones(X_net_exc.shape[1])

        # ----------------------------------------------------------------
        # Validation data preparation (if provided)
        # ----------------------------------------------------------------
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            q_val = self.intermediate_estimator_.predict(X_val)
            exceed_val = y_val > q_val
            if exceed_val.sum() >= 10:
                X_val_exc = X_val[exceed_val]
                z_val_exc = y_val[exceed_val] - q_val[exceed_val]
                q_val_exc = q_val[exceed_val]
                X_net_val = self._build_network_input(X_val_exc, q_val_exc)
                X_net_val = (X_net_val - self.feature_mean_) / self.feature_std_
                val_loader = self._make_dataloader(X_net_val, z_val_exc, shuffle=False)
            else:
                if self.verbose >= 1:
                    print(
                        "[EQRN] Fewer than 10 validation exceedances; "
                        "early stopping will use training loss."
                    )

        # ----------------------------------------------------------------
        # Step 2: Train GPD network
        # ----------------------------------------------------------------
        input_dim = X_net_exc.shape[1]
        self.network_ = GPDNet(
            input_dim=input_dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            p_drop=self.p_drop,
            shape_fixed=self.shape_fixed,
            batch_norm=self.batch_norm,
        ).to(self._device)

        if self.verbose >= 1:
            print(
                f"[EQRN] Step 2: training GPD network "
                f"(input_dim={input_dim}, params={self.network_.n_parameters}, "
                f"shape_fixed={self.shape_fixed})..."
            )

        effective_batch = int(min(self.batch_size, n_exceed))
        train_loader = self._make_dataloader(X_net_exc, z_exc, shuffle=True,
                                              batch_size=effective_batch)

        self._train(train_loader, val_loader)

        self._is_fitted = True

        # Warn if xi surface looks unusual
        with torch.no_grad():
            X_t = torch.tensor(X_net_exc, dtype=torch.float32, device=self._device)
            _, xi_t = self.network_(X_t)
        warn_xi_distribution(xi_t)

        if self.verbose >= 1:
            print(f"[EQRN] Training complete.")

        return self

    def _build_network_input(self, X: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Assemble network input from features and (optionally) threshold."""
        if self.append_quantile_feature:
            return np.column_stack([X, q])
        return X

    def _make_dataloader(
        self,
        X: np.ndarray,
        z: np.ndarray,
        shuffle: bool = True,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays."""
        bs = batch_size if batch_size is not None else self.batch_size
        X_t = torch.tensor(X, dtype=torch.float32)
        z_t = torch.tensor(z, dtype=torch.float32)
        ds = TensorDataset(X_t, z_t)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

    def _train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
    ) -> None:
        """Training loop with early stopping."""
        optimiser = optim.AdamW(
            self.network_.parameters(),
            lr=self.lr,
            weight_decay=self.l2_pen,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=self.patience // 2,
            min_lr=1e-6,
        )

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.train_losses_ = []
        self.val_losses_ = []

        for epoch in range(self.n_epochs):
            # Training pass
            self.network_.train()
            train_loss = self._run_epoch(train_loader, optimiser)
            self.train_losses_.append(train_loss)

            # Validation pass
            if val_loader is not None:
                self.network_.eval()
                with torch.no_grad():
                    val_loss = self._run_epoch(val_loader, optimiser=None)
                self.val_losses_.append(val_loss)
                monitor_loss = val_loss
            else:
                monitor_loss = train_loss

            scheduler.step(monitor_loss)

            # Early stopping
            if monitor_loss < best_loss - 1e-6:
                best_loss = monitor_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.network_.state_dict().items()}
            else:
                patience_counter += 1

            if self.verbose >= 2 and (epoch + 1) % 50 == 0:
                val_str = f", val_loss={monitor_loss:.4f}" if val_loader else ""
                print(f"  Epoch {epoch+1:4d}/{self.n_epochs}: train_loss={train_loss:.4f}{val_str}")

            if patience_counter >= self.patience:
                if self.verbose >= 1:
                    print(f"[EQRN] Early stopping at epoch {epoch+1} (patience={self.patience})")
                break

        # Restore best weights
        if best_state is not None:
            self.network_.load_state_dict(best_state)

    def _run_epoch(
        self,
        loader: DataLoader,
        optimiser: Optional[optim.Optimizer],
    ) -> float:
        """Run one epoch; return mean loss."""
        total_loss = 0.0
        n_batches = 0

        for X_batch, z_batch in loader:
            X_batch = X_batch.to(self._device)
            z_batch = z_batch.to(self._device)

            nu, xi = self.network_(X_batch)
            loss = ogpd_loss_tensor(z_batch, nu, xi)

            if self.shape_penalty > 0:
                shape_var_penalty = self.shape_penalty * xi.var()
                loss = loss + shape_var_penalty

            if optimiser is not None:
                optimiser.zero_grad()
                # Gradient clipping for stability
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), max_norm=1.0)
                optimiser.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before making predictions.")

    def _get_network_params(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (xi, sigma, threshold) for observations in X.

        Parameters
        ----------
        X:
            Covariate matrix, shape (n_test, n_features).

        Returns
        -------
        xi:
            Per-observation shape parameters.
        sigma:
            Per-observation scale parameters (in original units).
        threshold:
            Per-observation intermediate quantile thresholds.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        q = self.intermediate_estimator_.predict(X)
        X_net = self._build_network_input(X, q)
        X_net = (X_net - self.feature_mean_) / self.feature_std_

        X_t = torch.tensor(X_net, dtype=torch.float32, device=self._device)
        self.network_.eval()
        with torch.no_grad():
            nu, xi = self.network_(X_t)
        nu = nu.cpu().numpy()
        xi = xi.cpu().numpy()
        sigma = nu / (xi + 1.0)
        return xi, sigma, q

    def predict_params(self, X: np.ndarray) -> pd.DataFrame:
        """Return fitted GPD parameters per observation.

        Parameters
        ----------
        X:
            Covariate matrix, shape (n_test, n_features).

        Returns
        -------
        pd.DataFrame
            Columns: xi, sigma, nu, threshold.
        """
        xi, sigma, threshold = self._get_network_params(X)
        nu = sigma * (xi + 1.0)
        return pd.DataFrame({
            "xi": xi,
            "sigma": sigma,
            "nu": nu,
            "threshold": threshold,
        })

    def predict_quantile(self, X: np.ndarray, q: float = 0.995) -> np.ndarray:
        """Conditional quantile Q_x(q) at extreme level q.

        Parameters
        ----------
        X:
            Covariate matrix, shape (n_test, n_features).
        q:
            Quantile level. Must satisfy q > tau_0.

        Returns
        -------
        np.ndarray
            Conditional quantile per observation, shape (n_test,).
        """
        if q <= self.tau_0:
            raise ValueError(
                f"Quantile level q={q} must exceed tau_0={self.tau_0}. "
                "For quantiles below tau_0, use the intermediate quantile model directly."
            )
        xi, sigma, threshold = self._get_network_params(X)
        return eqrn_quantile(q, self.tau_0, threshold, xi, sigma)

    def predict_tvar(self, X: np.ndarray, q: float = 0.99) -> np.ndarray:
        """Conditional TVaR (Tail Value at Risk) at level q.

        TVaR_x(q) = E[Y | Y > Q_x(q), X = x].

        Parameters
        ----------
        X:
            Covariate matrix, shape (n_test, n_features).
        q:
            Tail probability level. Must satisfy q > tau_0.

        Returns
        -------
        np.ndarray
            Conditional TVaR per observation. Always >= predict_quantile(X, q).
        """
        if q <= self.tau_0:
            raise ValueError(
                f"TVaR level q={q} must exceed tau_0={self.tau_0}."
            )
        xi, sigma, threshold = self._get_network_params(X)
        return eqrn_tvar(q, self.tau_0, threshold, xi, sigma)

    def predict_exceedance_prob(
        self, X: np.ndarray, threshold: Union[float, np.ndarray]
    ) -> np.ndarray:
        """P(Y > threshold | X = x) for each observation.

        Parameters
        ----------
        X:
            Covariate matrix, shape (n_test, n_features).
        threshold:
            Fixed monetary threshold (scalar) or per-observation threshold
            array, shape (n_test,).

        Returns
        -------
        np.ndarray
            Exceedance probabilities, shape (n_test,).
        """
        xi, sigma, q_thresh = self._get_network_params(X)
        return eqrn_exceedance_prob(threshold, self.tau_0, q_thresh, xi, sigma)

    def predict_xl_layer(
        self,
        X: np.ndarray,
        attachment: Union[float, np.ndarray],
        limit: Union[float, np.ndarray],
        n_grid: int = 1000,
    ) -> np.ndarray:
        """Expected loss in XL layer (attachment xs limit) per risk.

        Computes E[min(Y - attachment, limit)^+ | X = x] via numerical
        integration over the GPD tail approximation.

        Parameters
        ----------
        X:
            Covariate matrix, shape (n_test, n_features).
        attachment:
            Layer attachment point(s). Scalar or shape (n_test,).
        limit:
            Layer limit(s). Scalar or shape (n_test,).
        n_grid:
            Integration grid points. More points = higher accuracy.

        Returns
        -------
        np.ndarray
            Expected layer loss per observation. Always >= 0.
        """
        xi, sigma, q_thresh = self._get_network_params(X)
        n = len(xi)

        attachment = np.broadcast_to(np.asarray(attachment, dtype=float), (n,)).copy()
        limit = np.broadcast_to(np.asarray(limit, dtype=float), (n,)).copy()

        return eqrn_xl_layer(
            attachment, limit, self.tau_0, q_thresh, xi, sigma, n_grid=n_grid
        )

    @property
    def is_fitted(self) -> bool:
        """True if fit() has been called."""
        return self._is_fitted
