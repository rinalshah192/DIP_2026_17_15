"""
Gabor filter-bank texture statistics.
ECE501 - Digital Image Processing (Group 15)

A bank of complex Gabor filters (Jain & Farrokhnia, 1991) tuned to the yarn
period: per (wavelength, orientation) the magnitude response is summarised by
its mean and standard deviation over the window. 3 wavelengths x 4 orientations
x 2 statistics = 24 values with the default config.
"""

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from fabric_defect.core.config import GaborConfig


def sigma_for(wavelength: float, bandwidth: float) -> float:
    """Gaussian sigma giving a half-response spatial-frequency bandwidth in octaves."""
    b = 2.0 ** bandwidth
    return wavelength / np.pi * np.sqrt(np.log(2) / 2) * (b + 1) / (b - 1)


class GaborFeatureExtractor:
    """Mean and std of the Gabor magnitude response per filter."""

    def __init__(self, cfg: GaborConfig = GaborConfig()):
        self.filters: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for lam in cfg.wavelengths:
            sigma = sigma_for(lam, cfg.bandwidth)
            ksize = 2 * int(np.ceil(3 * sigma)) + 1
            for deg in cfg.orientations_deg:
                theta = float(np.deg2rad(deg))
                even = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, cfg.gamma, 0, ktype=cv2.CV_32F)
                odd = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, cfg.gamma, np.pi / 2, ktype=cv2.CV_32F)
                even -= even.mean()  # zero DC: flat regions respond with 0, not with their brightness
                self.filters.append((f"l{lam:g}_o{deg:g}", even, odd))

    def responses(self, img: np.ndarray) -> np.ndarray:
        """Magnitude responses, shape (n_filters, H, W)."""
        f = img.astype(np.float32) / 255.0
        return np.stack([
            np.hypot(cv2.filter2D(f, -1, even, borderType=cv2.BORDER_REFLECT),
                     cv2.filter2D(f, -1, odd, borderType=cv2.BORDER_REFLECT))
            for _, even, odd in self.filters
        ])

    def extract_windows(self, img: np.ndarray, coords: Sequence[Tuple[int, int]], size: int) -> np.ndarray:
        """Features of square windows (top-left (y, x)) of img; responses are computed once."""
        resp = self.responses(img)
        n = len(self.filters)
        out = np.empty((len(coords), 2 * n), dtype=np.float32)
        for i, (y, x) in enumerate(coords):
            w = resp[:, y:y + size, x:x + size].reshape(n, -1)
            out[i, :n] = w.mean(axis=1)
            out[i, n:] = w.std(axis=1)
        return out

    def extract_batch(self, patches: np.ndarray) -> np.ndarray:
        size = patches.shape[-1] if len(patches) else 0
        return np.vstack([self.extract_windows(p, [(0, 0)], size) for p in patches]) if len(patches) \
            else np.empty((0, 2 * len(self.filters)), dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        names = [name for name, _, _ in self.filters]
        return [f"gabor_{n}_mean" for n in names] + [f"gabor_{n}_std" for n in names]
