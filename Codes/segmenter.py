"""
Defect Segmentation and Localization Module.
ECE501 - Digital Image Processing (Group 15)

Converts continuous texture-deviation fields into binary defect masks using:
1. Adaptive statistical thresholding (tau = mu + k * sigma)
2. Otsu's global thresholding baseline
3. Morphological filtering (opening, closing, small component suppression)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

from Codes.config import (
    ADAPTIVE_THRESHOLD_K,
    MIN_DEFECT_AREA_PIXELS
)


class DefectSegmenter:
    """Segments and localizes fabric surface defects from anomaly heatmaps."""

    def __init__(
        self,
        method: str = "adaptive",
        adaptive_k: float = ADAPTIVE_THRESHOLD_K,
        min_defect_area: int = MIN_DEFECT_AREA_PIXELS,
        kernel_size: int = 5
    ):
        self.method = method
        self.adaptive_k = adaptive_k
        self.min_defect_area = min_defect_area
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def threshold_heatmap(
        self,
        heatmap: np.ndarray,
        reference_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Threshold 2D deviation heatmap to generate raw candidate defect mask.

        Returns:
            Tuple of (raw_binary_mask_uint8, threshold_value_used)
        """
        if self.method == "adaptive" and reference_threshold is not None:
            thresh_val = float(reference_threshold)
            raw_mask = (heatmap >= thresh_val).astype(np.uint8) * 255
        elif self.method == "otsu":
            # Scale heatmap to [0, 255] uint8 for Otsu
            norm_hm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            otsu_val, raw_mask = cv2.threshold(norm_hm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_val = float(otsu_val)
        else:
            # Fallback to empirical statistical cutoff
            mu = float(np.mean(heatmap))
            sig = float(np.std(heatmap))
            thresh_val = mu + self.adaptive_k * sig
            raw_mask = (heatmap >= thresh_val).astype(np.uint8) * 255

        return raw_mask, thresh_val

    def apply_morphological_cleanup(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening to remove isolated noise specks,
        closing to bridge broken thread segments, and filter tiny blobs.
        """
        # 1. Opening: erosion followed by dilation (removes small background noise)
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.morph_kernel)

        # 2. Closing: dilation followed by erosion (closes gaps within defects)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel)

        # 3. Connected component area filtering
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        cleaned_mask = np.zeros_like(closed)

        for label in range(1, num_labels):  # Skip background (label 0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.min_defect_area:
                cleaned_mask[labels == label] = 255

        return cleaned_mask

    def segment(
        self,
        heatmap: np.ndarray,
        reference_threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Execute full segmentation and cleanup workflow.

        Returns:
            Dict containing raw_mask, cleaned_mask, and threshold_used.
        """
        raw_mask, thresh_val = self.threshold_heatmap(heatmap, reference_threshold)
        cleaned_mask = self.apply_morphological_cleanup(raw_mask)

        return {
            "raw_mask": raw_mask,
            "cleaned_mask": cleaned_mask,
            "threshold": thresh_val
        }
