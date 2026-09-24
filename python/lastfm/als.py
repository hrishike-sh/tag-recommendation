"""Implicit-feedback Matrix Factorization (MSVD / ALS) using confidence weighting.

Implements the Hu, Koren, Volinsky (2008) formulation:
    L = \\sum_{u,i} c_{ui} (p_{ui} - x_u^T y_i)^2 + \\lambda (\\sum_u ||x_u||^2 + \\sum_i ||y_i||^2)

Exploits sparsity via:
    C_u = I + D_u, where D_u is diagonal with nonzero entries d_{ui} = c_{ui} - 1.
    Y^T C_u Y = Y^T Y + Y^T D_u Y = Y^T Y + \\sum_{i \\in R_u} d_{ui} y_i y_i^T
    Y^T C_u p_u = Y^T (p_u + D_u p_u) = \\sum_{i \\in R_u} (1 + d_{ui}) p_{ui} y_i = \\sum_{i \\in R_u} c_{ui} p_{ui} y_i

Numerically stable linear solves are performed via scipy.linalg.cho_factor / cho_solve (or np.linalg.solve as fallback).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import time
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class ImplicitMSVD:
    """Implicit-feedback MSVD / ALS Recommender."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 15,
        seed: Optional[int] = 42,
        dtype: type = np.float32,
    ) -> None:
        """Initialize ImplicitMSVD model.

        Parameters
        ----------
        factors : int, default=64
            Number of latent factors (K).
        regularization : float, default=0.05
            L2 regularization penalty (lambda).
        iterations : int, default=15
            Number of ALS alternating update iterations.
        seed : int or None, default=42
            Random seed for reproducible latent factor initialization.
        dtype : type, default=np.float32
            Data type for latent factor matrices.
        """
        if factors <= 0:
            raise ValueError("factors must be a positive integer")
        if regularization < 0 or not math.isfinite(regularization):
            raise ValueError("regularization must be a nonnegative finite float")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed
        self.dtype = dtype

        self.user_factors: Optional[np.ndarray] = None  # Shape: (n_users, factors)
        self.item_factors: Optional[np.ndarray] = None  # Shape: (n_items, factors)
        self.n_users: int = 0
        self.n_items: int = 0
        self.history: List[dict[str, Any]] = []

    def _init_factors(self, n_users: int, n_items: int) -> None:
        """Initialize user and item factor matrices.

        Uses standard normal distribution scaled by 1/sqrt(factors) for item factors
        and zeros (or small scaled values) for user factors.
        """
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.factors)
        self.user_factors = rng.normal(0.0, scale, size=(n_users, self.factors)).astype(self.dtype)
        self.item_factors = rng.normal(0.0, scale, size=(n_items, self.factors)).astype(self.dtype)
        self.n_users = n_users
        self.n_items = n_items

    @staticmethod
    def _solve_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve A x = b using Cholesky decomposition with fallback to standard solve."""
        try:
            c, low = scipy.linalg.cho_factor(A, overwrite_a=False, check_finite=False)
            return scipy.linalg.cho_solve((c, low), b, overwrite_b=False, check_finite=False)
        except (scipy.linalg.LinAlgError, ValueError):
            return np.linalg.solve(A, b)

    def _update_users(
        self,
        pref_csr: sp.csr_matrix,
        conf_delta_csr: sp.csr_matrix,
        YtY: np.ndarray,
        reg_I: np.ndarray,
    ) -> None:
        """Update all user latent vectors x_u."""
        assert self.user_factors is not None
        assert self.item_factors is not None

        Y = self.item_factors
        K = self.factors
        d_data = conf_delta_csr.data.astype(self.dtype)
        same_structure = len(d_data) == len(pref_csr.data)

        for u in range(self.n_users):
            p_start, p_end = pref_csr.indptr[u], pref_csr.indptr[u + 1]
            if p_start == p_end:
                self.user_factors[u] = np.zeros(K, dtype=self.dtype)
                continue

            item_indices = pref_csr.indices[p_start:p_end]
            p_u = pref_csr.data[p_start:p_end]
            if same_structure:
                d_u = d_data[p_start:p_end]
            else:
                d_start, d_end = conf_delta_csr.indptr[u], conf_delta_csr.indptr[u + 1]
                if d_start == d_end:
                    d_u = np.zeros(len(item_indices), dtype=self.dtype)
                else:
                    d_indices = conf_delta_csr.indices[d_start:d_end]
                    d_sub = d_data[d_start:d_end]
                    if len(d_indices) == len(item_indices) and np.array_equal(d_indices, item_indices):
                        d_u = d_sub
                    else:
                        d_map = dict(zip(d_indices, d_sub))
                        d_u = np.array([d_map.get(idx, 0.0) for idx in item_indices], dtype=self.dtype)

            # Y_u shape: (num_user_items, K)
            Y_u = Y[item_indices]

            # A_u = Y^T Y + Y_u^T diag(d_u) Y_u + lambda I
            Y_u_weighted = Y_u * d_u[:, None]
            A_u = YtY + (Y_u_weighted.T @ Y_u) + reg_I

            # b_u = Y^T C_u p_u = \sum_{i \in R_u} (1 + d_{ui}) p_{ui} y_i
            c_u = 1.0 + d_u
            cp_u = (c_u * p_u.astype(self.dtype))[:, None]
            b_u = np.sum(Y_u * cp_u, axis=0)

            self.user_factors[u] = self._solve_cholesky(A_u, b_u)

    def _update_items(
        self,
        pref_csc: sp.csc_matrix,
        conf_delta_csc: sp.csc_matrix,
        XtX: np.ndarray,
        reg_I: np.ndarray,
    ) -> None:
        """Update all item latent vectors y_i."""
        assert self.user_factors is not None
        assert self.item_factors is not None

        X = self.user_factors
        K = self.factors
        d_data = conf_delta_csc.data.astype(self.dtype)
        same_structure = len(d_data) == len(pref_csc.data)

        for i in range(self.n_items):
            p_start, p_end = pref_csc.indptr[i], pref_csc.indptr[i + 1]
            if p_start == p_end:
                self.item_factors[i] = np.zeros(K, dtype=self.dtype)
                continue

            user_indices = pref_csc.indices[p_start:p_end]
            p_i = pref_csc.data[p_start:p_end]
            if same_structure:
                d_i = d_data[p_start:p_end]
            else:
                d_start, d_end = conf_delta_csc.indptr[i], conf_delta_csc.indptr[i + 1]
                if d_start == d_end:
                    d_i = np.zeros(len(user_indices), dtype=self.dtype)
                else:
                    d_indices = conf_delta_csc.indices[d_start:d_end]
                    d_sub = d_data[d_start:d_end]
                    if len(d_indices) == len(user_indices) and np.array_equal(d_indices, user_indices):
                        d_i = d_sub
                    else:
                        d_map = dict(zip(d_indices, d_sub))
                        d_i = np.array([d_map.get(idx, 0.0) for idx in user_indices], dtype=self.dtype)

            # Fast path for single-interaction items
            if len(user_indices) == 1:
                u_idx = user_indices[0]
                x_u = X[u_idx]
                d_val = float(d_i[0])
                p_val = float(p_i[0])
                A_i = XtX + (d_val * np.outer(x_u, x_u)) + reg_I
                b_i = (1.0 + d_val) * p_val * x_u
                self.item_factors[i] = self._solve_cholesky(A_i, b_i)
                continue

            # X_i shape: (num_item_users, K)
            X_i = X[user_indices]

            # A_i = X^T X + X_i^T diag(d_i) X_i + lambda I
            X_i_weighted = X_i * d_i[:, None]
            A_i = XtX + (X_i_weighted.T @ X_i) + reg_I

            # b_i = X^T C_i p_i = \sum_{u \in R_i} (1 + d_{ui}) p_{ui} x_u
            c_i = 1.0 + d_i
            cp_i = (c_i * p_i.astype(self.dtype))[:, None]
            b_i = np.sum(X_i * cp_i, axis=0)

            self.item_factors[i] = self._solve_cholesky(A_i, b_i)

    def compute_loss(
        self,
        pref_csr: sp.csr_matrix,
        conf_delta_csr: sp.csr_matrix,
    ) -> float:
        """Compute the exact Hu-Koren-Volinsky implicit ALS loss without dense matrices.

        L = \\sum_{u,i} c_{ui} (p_{ui} - x_u^T y_i)^2 + \\lambda (\\sum_u ||x_u||^2 + \\sum_i ||y_i||^2)
        Note that for unobserved entries, p_{ui}=0 and c_{ui}=1, so (p - x^T y)^2 = (x_u^T y_i)^2.
        Therefore:
          \\sum_{u,i} 1 * (x_u^T y_i)^2 = Tr((X^T X)(Y^T Y))
        For observed entries, we add:
          \\sum_{(u,i) \\in R} [ c_{ui} (p_{ui} - x_u^T y_i)^2 - (x_u^T y_i)^2 ]
        """
        assert self.user_factors is not None and self.item_factors is not None
        X = self.user_factors
        Y = self.item_factors

        XtX = X.T @ X
        YtY = Y.T @ Y

        # Base all-pairs loss at confidence 1: Tr(XtX @ YtY)
        base_loss = float(np.sum(XtX * YtY))

        # Correction for observed entries
        observed_correction = 0.0
        for u in range(self.n_users):
            start, end = pref_csr.indptr[u], pref_csr.indptr[u + 1]
            if start == end:
                continue
            item_indices = pref_csr.indices[start:end]
            p_u = pref_csr.data[start:end]
            
            d_start, d_end = conf_delta_csr.indptr[u], conf_delta_csr.indptr[u + 1]
            if d_start == d_end:
                d_u = np.zeros(len(item_indices), dtype=self.dtype)
            else:
                d_indices = conf_delta_csr.indices[d_start:d_end]
                d_data = conf_delta_csr.data[d_start:d_end]
                if len(d_indices) == len(item_indices) and np.array_equal(d_indices, item_indices):
                    d_u = d_data.astype(self.dtype)
                else:
                    d_map = dict(zip(d_indices, d_data))
                    d_u = np.array([d_map.get(idx, 0.0) for idx in item_indices], dtype=self.dtype)
            c_u = 1.0 + d_u

            preds = np.sum(X[u] * Y[item_indices], axis=1)
            # observed loss: c_ui * (p_ui - pred)^2
            obs_loss = c_u * ((p_u - preds) ** 2)
            # base loss included in Tr(XtX @ YtY): 1 * (0 - pred)^2 = pred^2
            base_included = preds ** 2
            observed_correction += float(np.sum(obs_loss - base_included))

        reg_loss = float(self.regularization * (np.sum(X ** 2) + np.sum(Y ** 2)))
        total_loss = base_loss + observed_correction + reg_loss
        return total_loss

    def fit(
        self,
        preferences: Union[sp.csr_matrix, sp.csc_matrix, sp.coo_matrix],
        confidence_delta: Union[sp.csr_matrix, sp.csc_matrix, sp.coo_matrix],
        compute_loss_history: bool = True,
        verbose: bool = True,
    ) -> "ImplicitMSVD":
        """Fit latent factor matrices X and Y using Alternating Least Squares.

        Parameters
        ----------
        preferences : sparse matrix
            Binary preferences matrix P (shape: n_users x n_items).
        confidence_delta : sparse matrix
            Confidence delta matrix D = C - 1 (shape: n_users x n_items).
        compute_loss_history : bool, default=True
            Whether to compute exact loss at each epoch.
        verbose : bool, default=True
            Whether to log training progress.
        """
        if preferences.shape != confidence_delta.shape:
            raise ValueError(
                f"Shape mismatch: preferences shape {preferences.shape} != confidence_delta shape {confidence_delta.shape}"
            )

        n_users, n_items = preferences.shape
        pref_csr = preferences.tocsr().astype(self.dtype)
        conf_delta_csr = confidence_delta.tocsr().astype(self.dtype)

        pref_csc = preferences.tocsc().astype(self.dtype)
        conf_delta_csc = confidence_delta.tocsc().astype(self.dtype)

        self._init_factors(n_users, n_items)
        assert self.user_factors is not None and self.item_factors is not None

        reg_I = (self.regularization * np.eye(self.factors, dtype=self.dtype))
        self.history = []

        total_start = time.perf_counter()

        for epoch in range(1, self.iterations + 1):
            epoch_start = time.perf_counter()

            # 1. Update users: requires Y^T Y
            YtY = self.item_factors.T @ self.item_factors
            self._update_users(pref_csr, conf_delta_csr, YtY, reg_I)

            # 2. Update items: requires X^T X
            XtX = self.user_factors.T @ self.user_factors
            self._update_items(pref_csc, conf_delta_csc, XtX, reg_I)

            epoch_time = time.perf_counter() - epoch_start

            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "elapsed_sec": epoch_time,
            }

            if compute_loss_history:
                loss = self.compute_loss(pref_csr, conf_delta_csr)
                epoch_record["loss"] = loss
                if verbose:
                    logger.info(
                        f"Epoch {epoch:3d}/{self.iterations:3d} - Loss: {loss:12.4f} - Time: {epoch_time:.3f}s"
                    )
            else:
                if verbose:
                    logger.info(
                        f"Epoch {epoch:3d}/{self.iterations:3d} - Time: {epoch_time:.3f}s"
                    )

            self.history.append(epoch_record)

        total_time = time.perf_counter() - total_start
        if verbose:
            logger.info(f"ALS training finished in {total_time:.2f}s across {self.iterations} iterations.")

        return self

    def recommend(
        self,
        user_index: int,
        k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[Union[Sequence[int], set[int], np.ndarray]] = None,
    ) -> List[Tuple[int, float]]:
        """Generate top-K recommendations for a given user.

        Parameters
        ----------
        user_index : int
            Zero-based user index.
        k : int, default=10
            Number of top items to recommend.
        exclude_seen : bool, default=True
            Whether to exclude already seen items.
        seen_items : sequence/set/ndarray or None
            Explicit collection of seen item indices.

        Returns
        -------
        list of (item_index, score) tuples sorted in descending order of score.
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model has not been fitted yet.")
        if user_index < 0 or user_index >= self.n_users:
            raise IndexError(f"user_index {user_index} out of bounds [0, {self.n_users})")
        if k <= 0:
            return []

        u_vec = self.user_factors[user_index]
        scores = u_vec @ self.item_factors.T  # Shape: (n_items,)

        if exclude_seen and seen_items is not None:
            scores = scores.copy()
            for idx in seen_items:
                if 0 <= idx < self.n_items:
                    scores[idx] = -np.inf

        num_candidates = min(k, self.n_items)
        if num_candidates == 0:
            return []

        # Argpartition for top-K followed by sorting top-K
        if num_candidates < self.n_items:
            candidate_indices = np.argpartition(scores, -num_candidates)[-num_candidates:]
            sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        else:
            sorted_indices = np.argsort(-scores)

        results: List[Tuple[int, float]] = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if np.isneginf(score):
                continue
            results.append((int(idx), score))
            if len(results) >= k:
                break

        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Save model factor matrices and configuration to a directory."""
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "user_factors.npy", self.user_factors)
        np.save(path / "item_factors.npy", self.item_factors)

        metadata = {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "seed": self.seed,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "dtype": str(np.dtype(self.dtype)),
            "history": self.history,
        }

        (path / "model_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "ImplicitMSVD":
        """Load a saved model from directory."""
        path = Path(directory)
        config_path = path / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found at {config_path}")

        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        dtype = np.dtype(metadata.get("dtype", "float32")).type

        model = cls(
            factors=metadata["factors"],
            regularization=metadata["regularization"],
            iterations=metadata["iterations"],
            seed=metadata.get("seed"),
            dtype=dtype,
        )

        model.user_factors = np.load(path / "user_factors.npy")
        model.item_factors = np.load(path / "item_factors.npy")
        model.n_users = metadata["n_users"]
        model.n_items = metadata["n_items"]
        model.history = metadata.get("history", [])

        return model
