"""
GLCM (Gray-Level Co-occurrence Matrix) Feature Extraction.
ECE501 - Digital Image Processing (Group 15)

Computes Haralick texture features across multiple pixel offsets and orientations:
- Contrast
- Dissimilarity
- Homogeneity
- Energy (Angular Second Moment)
- Correlation
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from skimage.feature import graycomatrix, graycoprops

from Codes.config import (
    GLCM_DISTANCES,
    GLCM_ANGLES,
    GLCM_LEVELS,
    GLCM_PROPERTIES
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
