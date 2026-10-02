"""
Pre-processing and Patch Extraction Module.
ECE501 - Digital Image Processing (Group 15)

Implements:
1. Grayscale loading and dynamic range normalisation
2. Noise filtering (Median and Gaussian)
3. Contrast normalisation (CLAHE and Global Histogram Equalisation)
4. Sliding-window patch extraction for both image strips and ground-truth masks
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from Codes.config import (
    MEDIAN_KERNEL_SIZE,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    PATCH_SIZE,
    PATCH_STRIDE
)


class Preprocessor:
    """Pre-processing engine for fabric defect inspection."""

    def __init__(
        self,
        use_median: bool = True,
        median_ksize: int = MEDIAN_KERNEL_SIZE,
        use_gaussian: bool = False,
        gaussian_ksize: Tuple[int, int] = GAUSSIAN_KERNEL_SIZE,
        gaussian_sigma: float = GAUSSIAN_SIGMA,
        use_clahe: bool = True,
        clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
        clahe_grid_size: Tuple[int, int] = CLAHE_GRID_SIZE,
        use_global_histeq: bool = False
    ):
        self.use_median = use_median
        self.median_ksize = median_ksize
        self.use_gaussian = use_gaussian
        self.gaussian_ksize = gaussian_ksize
        self.gaussian_sigma = gaussian_sigma
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.use_global_histeq = use_global_histeq

        # Initialize CLAHE operator
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )

    def load_image(self, path: str) -> np.ndarray:
        """Load image as grayscale uint8 array."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {path}")
        return img

    def load_mask(self, path: str, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load and binarize ground truth mask (0=normal, 255=defect)."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask at {path}")
        if target_shape is not None and mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        # Strict binarization
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask_bin

    def apply_filtering(self, img: np.ndarray) -> np.ndarray:
        """Apply noise suppression while preserving thread boundaries."""
        out = img.copy()
        if self.use_median:
            out = cv2.medianBlur(out, self.median_ksize)
        if self.use_gaussian:
            out = cv2.GaussianBlur(out, self.gaussian_ksize, self.gaussian_sigma)
        return out

    def apply_contrast_normalization(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE or global histogram equalization to normalize illumination gradient."""
        out = img.copy()
        if self.use_clahe:
            out = self._clahe.apply(out)
        elif self.use_global_histeq:
            out = cv2.equalizeHist(out)
        return out

    def process(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute full pre-processing stage.
        Returns dictionary containing intermediate and final processed stages.
        """
        raw = img.copy()
        filtered = self.apply_filtering(raw)
        normalized = self.apply_contrast_normalization(filtered)
        return {
            "raw": raw,
            "filtered": filtered,
            "normalized": normalized
        }


def extract_patches(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE
) -> Dict[str, any]:
    """
    Extract sliding window patches along the longitudinal fabric strip.

    Args:
        img: Pre-processed image strip (H x W), e.g. 4096 x 256
        mask: Optional paired binary ground-truth mask (H x W)
        patch_size: Size of square patch (default 256)
        stride: Step size along vertical axis (default 128 or 256)

    Returns:
        Dict with keys:
            'patches': List of patch arrays (patch_size x patch_size)
            'mask_patches': List of corresponding mask patches (if mask provided)
            'coordinates': List of (y, x) top-left coordinates
            'labels': List of binary labels (1=defective patch, 0=defect-free)
    """
    H, W = img.shape[:2]
    patches = []
    mask_patches = []
    coordinates = []
    labels = []

    # Slide vertically along the long dimension (and horizontally if W > patch_size)
    y_starts = list(range(0, H - patch_size + 1, stride))
    if y_starts[-1] != H - patch_size:
        y_starts.append(H - patch_size)  # Ensure bottom edge is covered

    x_starts = list(range(0, max(1, W - patch_size + 1), stride))
    if len(x_starts) == 0:
        x_starts = [0]

    for y in y_starts:
        for x in x_starts:
            patch = img[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            coordinates.append((y, x))

            if mask is not None:
                m_patch = mask[y : y + patch_size, x : x + patch_size]
                mask_patches.append(m_patch)
                # Any patch containing > 25 defective pixels is labelled defective (1)
                is_defective = int(np.count_nonzero(m_patch) > 25)
                labels.append(is_defective)

    return {
        "patches": np.array(patches),
        "mask_patches": np.array(mask_patches) if mask is not None else None,
        "coordinates": coordinates,
        "labels": np.array(labels) if mask is not None else None
    }
