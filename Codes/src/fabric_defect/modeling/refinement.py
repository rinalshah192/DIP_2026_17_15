"""
Coarse-to-fine localization.
ECE501 - Digital Image Processing (Group 15)

256x256 windows give stable texture statistics but flag 65,536 px for a thread
that covers a few hundred, which caps pixel IoU near zero. Refinement keeps
the 256 decision as a region of interest and re-scores the 64x64 windows that
lie entirely inside it against a second, 64x64 reference model of the same
fabric. Only flagged 64x64 windows reach the predicted mask. With several
levels (e.g. 64 then 32) each level's flagged windows bound the next level.
"""

import numpy as np

from fabric_defect.features.preprocessing import window_sums


def coarse_region(flags: np.ndarray, coords: np.ndarray, shape, size: int) -> np.ndarray:
    """Boolean mask covering every flagged coarse window."""
    region = np.zeros(shape, dtype=bool)
    for (y, x) in coords[np.asarray(flags, dtype=bool)]:
        region[y:y + size, x:x + size] = True
    return region


def windows_inside(region: np.ndarray, coords: np.ndarray, size: int) -> np.ndarray:
    """Which fine windows lie entirely inside the region (summed-area table, O(1) per window)."""
    return window_sums(region, coords, size) == size * size


def refined_mask(region: np.ndarray, coords: np.ndarray, scores: np.ndarray, tau: float, size: int) -> np.ndarray:
    """uint8 mask (0/255) of fine windows inside the region whose score reaches tau."""
    mask = np.zeros(region.shape, dtype=np.uint8)
    hit = windows_inside(region, coords, size) & (scores >= tau)
    for (y, x) in coords[hit]:
        mask[y:y + size, x:x + size] = 255
    return mask
