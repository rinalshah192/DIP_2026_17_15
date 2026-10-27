"""
Multi-scale rotation-invariant uniform LBP histograms.
ECE501 - Digital Image Processing (Group 15)
"""

from typing import List, Sequence, Tuple

import numpy as np
from skimage.feature import local_binary_pattern

from fabric_defect.core.config import LBPConfig


class LBPFeatureExtractor:
    """Extracts multi-scale uniform rotation-invariant LBP histogram descriptors."""

    def __init__(self, cfg: LBPConfig = LBPConfig()):
        self.scales = [tuple(s) for s in cfg.scales]

    def extract_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale LBP normalized histogram for a single patch.

        Args:
            patch: Grayscale 2D array (H x W)

        Returns:
            1D concatenated histogram vector
        """
        hist_parts = []
        for P, R in self.scales:
            lbp_map = local_binary_pattern(patch, P=P, R=R, method="uniform")
            # Uniform rotation-invariant (riu2) mapping yields P + 2 codes
            n_bins = P + 2

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

    def extract_windows(self, img: np.ndarray, coords: Sequence[Tuple[int, int]], size: int) -> np.ndarray:
        """Histograms of square windows (top-left (y, x)) of img.

        The LBP codes are computed once for the whole image, so overlapping
        windows share the work and border codes see their true neighbours.
        """
        maps = [(local_binary_pattern(img, P=P, R=R, method="uniform").astype(np.int64), P + 2)
                for P, R in self.scales]
        out = np.empty((len(coords), sum(n for _, n in maps)), dtype=np.float32)
        for i, (y, x) in enumerate(coords):
            out[i] = np.concatenate([
                np.bincount(m[y:y + size, x:x + size].ravel(), minlength=n) / float(size * size)
                for m, n in maps
            ])
        return out

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
        for P, R in self.scales:
            for b in range(P + 2):
                names.append(f"lbp_P{P}_R{R}_bin{b}")
        return names
