"""
Patch texture descriptors.
ECE501 - Digital Image Processing (Group 15)

1. GLCM (Haralick) statistics: contrast, dissimilarity, homogeneity, energy,
   correlation; mean and std across orientations per distance.
2. Multi-scale rotation-invariant uniform LBP histograms.
"""

from typing import Dict, List, Optional

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from fabric_defect.config import (
    GLCM_DISTANCES,
    GLCM_ANGLES,
    GLCM_LEVELS,
    GLCM_PROPERTIES,
    LBP_CONFIGS
)


class GLCMFeatureExtractor:
    """Extracts statistical texture descriptors from Gray-Level Co-occurrence Matrices."""

    def __init__(
        self,
        distances: List[int] = GLCM_DISTANCES,
        angles: List[float] = GLCM_ANGLES,
        levels: int = GLCM_LEVELS,
        properties: List[str] = GLCM_PROPERTIES
    ):
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.properties = properties

    def extract_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract GLCM feature vector for a single 2D image patch.

        Args:
            patch: Grayscale 2D array (H x W), values in [0, 255]

        Returns:
            1D numpy array containing concatenated Haralick statistics
            averaged across orientations for rotational invariance,
            plus orientation-specific values for directional defect sensitivity.
        """
        # Quantize to specified number of grey levels
        if self.levels < 256:
            p_min = float(patch.min())
            p_max = float(patch.max())
            if p_max - p_min > 5:
                norm_p = (patch.astype(np.float32) - p_min) / (p_max - p_min) * (self.levels - 1)
                quantized = np.clip(np.round(norm_p), 0, self.levels - 1).astype(np.uint8)
            else:
                quantized = (patch.astype(np.float32) / 256.0 * self.levels).astype(np.uint8)
                quantized = np.clip(quantized, 0, self.levels - 1)
        else:
            quantized = patch.astype(np.uint8)

        # Compute full co-occurrence matrix
        glcm = graycomatrix(
            quantized,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=True,
            normed=True
        )

        features = []
        for prop in self.properties:
            val_matrix = graycoprops(glcm, prop)  # Shape: (len(distances), len(angles))
            # Average across angles for rotation-invariant feature
            mean_over_angles = val_matrix.mean(axis=1)
            # Std deviation across angles (directional anisotropy)
            std_over_angles = val_matrix.std(axis=1)

            features.extend(mean_over_angles.tolist())
            features.extend(std_over_angles.tolist())

        return np.array(features, dtype=np.float32)

    def extract_batch(self, patches: np.ndarray) -> np.ndarray:
        """
        Extract GLCM feature matrix for a batch of patches.

        Args:
            patches: Array of shape (N, H, W)

        Returns:
            Feature matrix of shape (N, D_glcm)
        """
        feat_list = [self.extract_patch(p) for p in patches]
        return np.array(feat_list, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Return names of extracted GLCM features for explainability."""
        names = []
        for prop in self.properties:
            for d in self.distances:
                names.append(f"glcm_{prop}_d{d}_mean")
                names.append(f"glcm_{prop}_d{d}_std")
        return names


class LBPFeatureExtractor:
    """Extracts multi-scale uniform rotation-invariant LBP histogram descriptors."""

    def __init__(self, configs: Optional[List[Dict]] = None):
        self.configs = configs if configs is not None else LBP_CONFIGS

    def extract_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale LBP normalized histogram for a single patch.

        Args:
            patch: Grayscale 2D array (H x W)

        Returns:
            1D concatenated histogram vector
        """
        hist_parts = []
        for cfg in self.configs:
            P = cfg["points"]
            R = cfg["radius"]
            method = cfg.get("method", "uniform")

            # Compute LBP map
            lbp_map = local_binary_pattern(patch, P=P, R=R, method=method)

            # Uniform rotation-invariant mapping yields (P + 2) bins
            n_bins = P + 2 if method in ["uniform", "ror", "riu2"] else int(2**P)

            hist, _ = np.histogram(
                lbp_map.ravel(),
                bins=n_bins,
                range=(0, n_bins),
                density=True
            )
            # Ensure L1 normalization
            norm_hist = hist / (np.sum(hist) + 1e-7)
            hist_parts.append(norm_hist)

        return np.concatenate(hist_parts).astype(np.float32)

    def extract_batch(self, patches: np.ndarray) -> np.ndarray:
        """
        Extract LBP feature matrix for a batch of patches.

        Args:
            patches: Array of shape (N, H, W)

        Returns:
            Feature matrix of shape (N, D_lbp)
        """
        feat_list = [self.extract_patch(p) for p in patches]
        return np.array(feat_list, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Return descriptive names of LBP histogram bins."""
        names = []
        for cfg in self.configs:
            P = cfg["points"]
            R = cfg["radius"]
            n_bins = P + 2
            for b in range(n_bins):
                names.append(f"lbp_P{P}_R{R}_bin{b}")
        return names
