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
from fabric_defect.config import (
    MEDIAN_KERNEL_SIZE,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    PATCH_SIZE,
    PATCH_STRIDE,
    SELVAGE_MARGIN_PIXELS,
    USE_ILLUMINATION_FLATTENING,
    ILLUMINATION_BLUR_SIGMA
)


def detect_fabric_roi(img: np.ndarray, margin: int = SELVAGE_MARGIN_PIXELS) -> Tuple[int, int]:
    """
    Detect the active woven fabric region, excluding uncropped white conveyor
    margins and frayed selvage borders along the line-scan strip.

    Returns:
        (col_start, col_end): Valid horizontal span of the genuine fabric.
    """
    col_means = img.mean(axis=0)
    col_stds = img.std(axis=0)

    # Valid fabric columns have intensity well below white background (255)
    # and non-zero texture variance
    valid_cols = np.where((col_means < 245) & (col_stds > 2.0))[0]
    if len(valid_cols) == 0:
        return 0, img.shape[1]

    col_start = max(0, int(valid_cols[0]) + margin)
    col_end = min(img.shape[1], int(valid_cols[-1]) - margin + 1)

    # Ensure minimum width covers at least one full patch
    if col_end - col_start < PATCH_SIZE:
        col_start = max(0, int(valid_cols[0]))
        col_end = min(img.shape[1], int(valid_cols[-1]) + 1)
        if col_end - col_start < PATCH_SIZE:
            return 0, img.shape[1]

    return col_start, col_end


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
        use_global_histeq: bool = False,
        use_illumination_flattening: bool = USE_ILLUMINATION_FLATTENING,
        illumination_sigma: float = ILLUMINATION_BLUR_SIGMA
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
        self.use_illumination_flattening = use_illumination_flattening
        self.illumination_sigma = illumination_sigma

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

    def apply_illumination_flattening(self, img: np.ndarray) -> np.ndarray:
        """Suppresses low-frequency illumination gradients and lens vignetting."""
        if not self.use_illumination_flattening:
            return img
        bg = cv2.GaussianBlur(img, (0, 0), sigmaX=self.illumination_sigma, sigmaY=self.illumination_sigma)
        flattened = cv2.addWeighted(img, 1.0, bg, -1.0, 128)
        return flattened

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
        flattened = self.apply_illumination_flattening(raw)
        filtered = self.apply_filtering(flattened)
        normalized = self.apply_contrast_normalization(filtered)
        return {
            "raw": raw,
            "flattened": flattened,
            "filtered": filtered,
            "normalized": normalized
        }


def extract_patches(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE,
    offset_x: int = 0,
    offset_y: int = 0
) -> Dict[str, any]:
    """
    Extract sliding window patches along the longitudinal fabric strip.

    Args:
        img: Pre-processed image strip (H x W), e.g. 256 x 4096 or cropped ROI
        mask: Optional paired binary ground-truth mask (H x W)
        patch_size: Size of square patch (default 256)
        stride: Step size along horizontal/vertical axes (default 128 or 256)
        offset_x: Horizontal coordinate offset if img was cropped from larger strip
        offset_y: Vertical coordinate offset if img was cropped from larger strip

    Returns:
        Dict with keys:
            'patches': List of patch arrays (patch_size x patch_size)
            'mask_patches': List of corresponding mask patches (if mask provided)
            'coordinates': List of (y, x) top-left coordinates in reference strip space
            'labels': List of binary labels (1=defective patch, 0=defect-free)
    """
    H, W = img.shape[:2]
    patches = []
    mask_patches = []
    coordinates = []
    labels = []

    # Slide vertically along the long dimension (and horizontally if W > patch_size)
    y_starts = list(range(0, max(1, H - patch_size + 1), stride))
    if y_starts[-1] != H - patch_size and H >= patch_size:
        y_starts.append(H - patch_size)  # Ensure bottom edge is covered

    x_starts = list(range(0, max(1, W - patch_size + 1), stride))
    if x_starts[-1] != W - patch_size and W >= patch_size:
        x_starts.append(W - patch_size)  # Ensure right edge is covered

    for y in y_starts:
        for x in x_starts:
            patch = img[y : y + patch_size, x : x + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            patches.append(patch)
            coordinates.append((y + offset_y, x + offset_x))

            if mask is not None:
                m_patch = mask[y : y + patch_size, x : x + patch_size]
                mask_patches.append(m_patch)
                # Any patch containing > 25 defective pixels is labelled defective (1)
                is_defective = int(np.count_nonzero(m_patch) > 25)
                labels.append(is_defective)

    return {
        "patches": np.array(patches) if patches else np.empty((0, patch_size, patch_size), dtype=np.uint8),
        "mask_patches": np.array(mask_patches) if mask is not None and mask_patches else None,
        "coordinates": coordinates,
        "labels": np.array(labels) if mask is not None and labels else np.array([])
    }
