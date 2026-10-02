"""
Local Binary Patterns (LBP) Feature Extraction.
ECE501 - Digital Image Processing (Group 15)

Implements:
1. Multi-scale rotation-invariant uniform LBP (P=8, R=1 and P=16, R=2)
2. Normalized spatial histogram feature vectors for patch characterization
"""

import numpy as np
from typing import List, Dict, Optional
from skimage.feature import local_binary_pattern

from Codes.config import LBP_CONFIGS


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
