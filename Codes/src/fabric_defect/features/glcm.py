"""
GLCM (Haralick) texture statistics.
ECE501 - Digital Image Processing (Group 15)

Contrast, dissimilarity, homogeneity, energy and correlation, as mean and
std across orientations for each pixel distance.
"""

from typing import List, Sequence, Tuple

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from fabric_defect.core.config import GLCMConfig


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

    def extract_windows(self, img: np.ndarray, coords: Sequence[Tuple[int, int]], size: int) -> np.ndarray:
        """Features of square windows (top-left (y, x)) of img."""
        out = [self.extract_patch(img[y:y + size, x:x + size]) for y, x in coords]
        return np.array(out, dtype=np.float32).reshape(len(coords), -1)

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
