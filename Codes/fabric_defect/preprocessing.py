"""
Pre-processing and Patch Extraction Module.
ECE501 - Digital Image Processing (Group 15)

Implements:
1. Fabric ROI detection (drops conveyor background and frayed selvage columns)
2. High-pass illumination flattening
3. Noise filtering (median, optional Gaussian)
4. Contrast normalisation (CLAHE or global histogram equalisation)
5. Sliding-window patch extraction for image strips and ground-truth masks
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from fabric_defect.config import PreprocessConfig


def load_image(path: str) -> np.ndarray:
    """Load image as grayscale uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {path}")
    return img


def detect_fabric_roi(img: np.ndarray, margin: int, min_width: int) -> Tuple[int, int]:
    """
    Detect the active woven fabric region, excluding uncropped white conveyor
    margins and frayed selvage borders along the line-scan strip.

    Returns:
        (col_start, col_end): Valid horizontal span of the genuine fabric.
    """
    col_means = img.mean(axis=0)
    col_stds = img.std(axis=0)

    # Valid fabric columns are darker than the white background and carry texture
    valid_cols = np.where((col_means < 245) & (col_stds > 2.0))[0]
    if len(valid_cols) == 0:
        return 0, img.shape[1]

    col_start = max(0, int(valid_cols[0]) + margin)
    col_end = min(img.shape[1], int(valid_cols[-1]) - margin + 1)

    # Ensure the ROI still holds at least one full patch
    if col_end - col_start < min_width:
        col_start = max(0, int(valid_cols[0]))
        col_end = min(img.shape[1], int(valid_cols[-1]) + 1)
        if col_end - col_start < min_width:
            return 0, img.shape[1]

    return col_start, col_end


class Preprocessor:
    """Pre-processing engine for fabric defect inspection."""

    def __init__(self, cfg: PreprocessConfig = PreprocessConfig()):
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=tuple(cfg.clahe_grid))

    def apply_illumination_flattening(self, img: np.ndarray) -> np.ndarray:
        """Suppress low-frequency illumination gradients and lens vignetting."""
        sigma = self.cfg.illumination_sigma
        if not sigma:
            return img
        bg = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return cv2.addWeighted(img, 1.0, bg, -1.0, 128)

    def apply_filtering(self, img: np.ndarray) -> np.ndarray:
        """Apply noise suppression while preserving thread boundaries."""
        out = img
        if self.cfg.median_ksize:
            out = cv2.medianBlur(out, self.cfg.median_ksize)
        if self.cfg.gaussian_ksize:
            k = self.cfg.gaussian_ksize
            out = cv2.GaussianBlur(out, (k, k), self.cfg.gaussian_sigma)
        return out

    def apply_contrast_normalization(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE or global histogram equalisation."""
        if self.cfg.contrast == "clahe":
            return self._clahe.apply(img)
        if self.cfg.contrast == "histeq":
            return cv2.equalizeHist(img)
        return img

    def process(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the full pre-processing chain and return every intermediate stage."""
        flattened = self.apply_illumination_flattening(img)
        filtered = self.apply_filtering(flattened)
        normalized = self.apply_contrast_normalization(filtered)
        return {
            "raw": img,
            "flattened": flattened,
            "filtered": filtered,
            "normalized": normalized
        }


def _window_starts(length: int, size: int, stride: int) -> list:
    """Window start offsets that cover [0, length) and always include the last full window."""
    if length < size:
        return []
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def extract_patches(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    patch_size: int = 256,
    stride: int = 128,
    offset_x: int = 0,
    offset_y: int = 0,
    defect_min_pixels: int = 25
) -> Dict[str, Any]:
    """
    Extract sliding-window patches along the fabric strip.

    Args:
        img: Pre-processed image strip (H x W), e.g. 256 x 4096 or a cropped ROI
        mask: Optional paired binary ground-truth mask (H x W)
        patch_size: Side of the square patch
        stride: Step between windows along both axes
        offset_x, offset_y: Offset of img inside the full strip (if it was cropped)
        defect_min_pixels: A patch is labelled defective when its mask has more pixels than this

    Returns:
        Dict with 'patches' (N, s, s), 'coordinates' [(y, x)] in full-strip space
        and 'labels' (N,) when a mask is given.
    """
    H, W = img.shape[:2]
    patches, coordinates, labels = [], [], []

    for y in _window_starts(H, patch_size, stride):
        for x in _window_starts(W, patch_size, stride):
            patches.append(img[y:y + patch_size, x:x + patch_size])
            coordinates.append((y + offset_y, x + offset_x))
            if mask is not None:
                m_patch = mask[y:y + patch_size, x:x + patch_size]
                labels.append(int(np.count_nonzero(m_patch) > defect_min_pixels))

    return {
        "patches": np.array(patches) if patches else np.empty((0, patch_size, patch_size), dtype=img.dtype),
        "coordinates": coordinates,
        "labels": np.array(labels, dtype=int)
    }
