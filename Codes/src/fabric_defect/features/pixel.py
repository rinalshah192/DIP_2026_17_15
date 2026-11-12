"""
Dense per-pixel filter-bank features.
ECE501 - Digital Image Processing (Group 15)

Window descriptors say *where* a defect is to within a window; thin AFID
defects (broken ends and picks are 1-6 px wide) need a per-pixel decision.
Every pixel of the pre-processed fabric ROI gets classical filter responses:

- intensity, and its deviation from median filters of 5, 15 and 31 px
- local standard deviation over 5, 15 and 31 px
- weft-line and warp-line responses: deviation from each column's median
  (a missing weft yarn is a dark or bright row) and from each row's median
  (a missing warp yarn is a column), raw and smoothed *along* the yarn so a
  line that persists over many pixels stands out from texture noise
- Laplacian of Gaussian at sigma 1, 2 and 4 px (blobs, knots, neps)
- smoothed gradient magnitude along x and y
- the 12 Gabor magnitude responses of the window descriptor bank

Each channel is robust-z-scored within the strip (median / MAD), which removes
the strip-wide offsets between AFID's defect-free and defective captures.
"""

from typing import List

import cv2
import numpy as np

from fabric_defect.features.gabor import GaborFeatureExtractor


def _robust(a: np.ndarray) -> np.ndarray:
    med = np.median(a)
    return (a - med) / (np.median(np.abs(a - med)) + 1e-6)


class PixelFeatureExtractor:
    """(H, W, F) feature stack for a pre-processed uint8 strip; F = 30."""

    def __init__(self):
        self.gabor = GaborFeatureExtractor()

    def extract(self, img: np.ndarray) -> np.ndarray:
        f = img.astype(np.float32)
        ch = [f]
        for k in (5, 15, 31):
            ch.append(f - cv2.medianBlur(img, k).astype(np.float32))
            mu = cv2.blur(f, (k, k))
            ch.append(np.sqrt(np.maximum(cv2.blur(f * f, (k, k)) - mu * mu, 0)))
        weft = f - np.median(f, axis=0, keepdims=True)  # a defective row stands out within its column
        warp = f - np.median(f, axis=1, keepdims=True)  # a defective column stands out within its row
        ch += [weft, warp]
        for k in (9, 33):  # cv2 ksize is (width, height): smooth weft lines along x, warp lines along y
            ch += [cv2.blur(weft, (k, 1)), cv2.blur(warp, (1, k))]
        for s in (1.0, 2.0, 4.0):
            ch.append(cv2.Laplacian(cv2.GaussianBlur(f, (0, 0), s), cv2.CV_32F))
        ch.append(cv2.blur(np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0)), (7, 7)))
        ch.append(cv2.blur(np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1)), (7, 7)))
        ch.extend(self.gabor.responses(img))
        return np.stack([_robust(c) for c in ch], axis=-1).astype(np.float32)

    def get_feature_names(self) -> List[str]:
        names = ["intensity"]
        for k in (5, 15, 31):
            names += [f"median_dev_{k}", f"local_std_{k}"]
        names += ["weft_line", "warp_line"]
        for k in (9, 33):
            names += [f"weft_line_smooth{k}", f"warp_line_smooth{k}"]
        names += [f"log_sigma{s:g}" for s in (1, 2, 4)] + ["grad_x", "grad_y"]
        names += [f"gabor_{n}" for n, _, _ in self.gabor.filters]
        return names
