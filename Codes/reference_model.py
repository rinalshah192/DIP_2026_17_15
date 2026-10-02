"""
Reference Normal Texture Modeling and Deviation Scoring.
ECE501 - Digital Image Processing (Group 15)

Constructs statistical models of defect-free textile surfaces
and calculates continuous texture-deviation maps for anomaly localization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import mahalanobis, cdist

from Codes.config import (
    MAHALANOBIS_REGULARIZATION,
    ADAPTIVE_THRESHOLD_K
)


class TextureReferenceModel:
    """
    Models the distribution of normal, defect-free fabric texture
    and computes statistical anomaly deviation scores for query patches.
    """

    def __init__(
        self,
        fabric_type: str,
        metric: str = "mahalanobis",
        regularization: float = MAHALANOBIS_REGULARIZATION,
        adaptive_k: float = ADAPTIVE_THRESHOLD_K
    ):
        self.fabric_type = fabric_type
        self.metric = metric
        self.regularization = regularization
        self.adaptive_k = adaptive_k

        self.mean_vector: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None
        self.inv_covariance: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        # Baseline statistics on training normal patches
        self.normal_score_mean: float = 0.0
        self.normal_score_std: float = 1.0
        self.adaptive_threshold: float = 2.5
        self.is_fitted: bool = False

    def fit(self, X_normal: np.ndarray) -> "TextureReferenceModel":
        """
        Fit reference model on feature vectors extracted from defect-free patches.

        Args:
            X_normal: Array of shape (N, D) from clean fabric samples.
        """
        if len(X_normal) == 0:
            raise ValueError("Cannot fit reference model with empty feature array.")

        N, D = X_normal.shape
        self.mean_vector = np.mean(X_normal, axis=0)
        self.feature_std = np.std(X_normal, axis=0) + 1e-6

        # Regularized Covariance
        raw_cov = np.cov(X_normal, rowvar=False) if N > 1 else np.eye(D)
        if raw_cov.ndim == 0:
            raw_cov = np.array([[raw_cov]])

        # Add Tikhonov regularization along diagonal for numerical stability
        reg_cov = raw_cov + self.regularization * np.eye(D) * np.trace(raw_cov) / D
        self.covariance_matrix = reg_cov

        try:
            self.inv_covariance = np.linalg.pinv(reg_cov)
        except Exception:
            # Fallback to diagonal inverse if pinv encounters conditioning issues
            self.inv_covariance = np.diag(1.0 / (np.var(X_normal, axis=0) + 1e-4))

        self.is_fitted = True

        # Evaluate baseline scores on the reference training set
        scores = self.score_batch(X_normal)
        self.normal_score_mean = float(np.mean(scores))
        self.normal_score_std = float(np.std(scores)) + 1e-6
        self.adaptive_threshold = self.normal_score_mean + self.adaptive_k * self.normal_score_std

        return self

    def score_patch(self, x: np.ndarray) -> float:
        """Compute anomaly deviation score for a single feature vector."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring patches.")

        delta = x - self.mean_vector

        if self.metric == "mahalanobis":
            dist = np.sqrt(np.maximum(0.0, np.dot(np.dot(delta, self.inv_covariance), delta)))
        elif self.metric == "euclidean_norm":
            # Normalized Euclidean (z-score distance)
            dist = np.sqrt(np.sum((delta / self.feature_std) ** 2))
        else:
            dist = np.linalg.norm(delta)

        return float(dist)

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for a matrix of query patches."""
        scores = [self.score_patch(x) for x in X]
        return np.array(scores, dtype=np.float32)

    def generate_deviation_map(
        self,
        patch_scores: np.ndarray,
        coordinates: List[Tuple[int, int]],
        strip_shape: Tuple[int, int],
        patch_size: int = 256
    ) -> np.ndarray:
        """
        Reconstruct a 2D continuous deviation heatmap across the full fabric strip.

        Args:
            patch_scores: 1D array of anomaly scores per patch
            coordinates: Top-left (y, x) position of each patch
            strip_shape: (Height, Width) of the strip, e.g. (4096, 256)
            patch_size: Size of patch (256)

        Returns:
            2D float32 array (H x W) containing the spatial anomaly heatmap
        """
        H, W = strip_shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        weight_map = np.zeros((H, W), dtype=np.float32)

        for score, (y, x) in zip(patch_scores, coordinates):
            y_end = min(H, y + patch_size)
            x_end = min(W, x + patch_size)

            heatmap[y:y_end, x:x_end] += score
            weight_map[y:y_end, x:x_end] += 1.0

        # Average overlapping areas (e.g. from 50% stride)
        weight_map[weight_map == 0] = 1.0
        normalized_heatmap = heatmap / weight_map
        return normalized_heatmap
