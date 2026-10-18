"""
Patch texture descriptors.
ECE501 - Digital Image Processing (Group 15)

1. GLCM (Haralick) statistics: contrast, dissimilarity, homogeneity, energy,
   correlation; mean and std across orientations per distance.
2. Multi-scale rotation-invariant uniform LBP histograms.
"""

from typing import List

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from fabric_defect.config import GLCMConfig, LBPConfig


class GLCMFeatureExtractor:
    """Extracts statistical texture descriptors from Gray-Level Co-occurrence Matrices."""

    def __init__(self, cfg: GLCMConfig = GLCMConfig()):
        self.distances = list(cfg.distances)
        self.angles = [float(np.deg2rad(a)) for a in cfg.angles_deg]
        self.levels = cfg.levels
        self.properties = list(cfg.properties)

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
        # Same order as extract_patch: per property, all means then all stds
        names = []
        for prop in self.properties:
            names += [f"glcm_{prop}_d{d}_mean" for d in self.distances]
            names += [f"glcm_{prop}_d{d}_std" for d in self.distances]
        return names


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


class TextureDescriptor:
    """Fused GLCM + LBP patch descriptor (D = 30 + 28 = 58 with the default config)."""

    def __init__(self, glcm: GLCMConfig = GLCMConfig(), lbp: LBPConfig = LBPConfig()):
        self.glcm = GLCMFeatureExtractor(glcm)
        self.lbp = LBPFeatureExtractor(lbp)

    def extract_batch(self, patches: np.ndarray) -> np.ndarray:
        return np.hstack([self.glcm.extract_batch(patches), self.lbp.extract_batch(patches)])

    def get_feature_names(self) -> List[str]:
        return self.glcm.get_feature_names() + self.lbp.get_feature_names()
