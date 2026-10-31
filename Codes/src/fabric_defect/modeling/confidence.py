"""
Image-level decision and confidence.
ECE501 - Digital Image Processing (Group 15)

A strip's image score is its most anomalous window, normalised by its fabric's
threshold (S = max_i s_i / tau_f), so that scores of different fabrics share one
scale. Its confidence is a conformal p-value (Vovk et al., 2005) against the
image scores of held-out defect-free calibration strips:

    p = (1 + #{calibration S >= S}) / (n + 1),   confidence = 1 - p,

and the strip is called defective when p <= alpha. Under exchangeability of
defect-free strips this bounds the image-level false-alarm rate by alpha.
"""

import numpy as np


def image_score(scores: np.ndarray, tau: float) -> float:
    """Most anomalous window relative to the threshold; >= 1 means at least one window is flagged."""
    if len(scores) == 0:
        return float("nan")
    return float(np.max(scores) / tau) if tau > 0 else float(np.max(scores))


class ConformalConfidence:
    """p-values of image scores against defect-free calibration strips."""

    def __init__(self, calibration_scores):
        cal = np.asarray([s for s in calibration_scores if np.isfinite(s)], dtype=np.float64)
        if len(cal) == 0:
            raise ValueError("Conformal calibration needs at least one defect-free strip.")
        self.cal = np.sort(cal)

    def p_value(self, s: float) -> float:
        # count of calibration scores >= s, via the sorted array
        at_least = len(self.cal) - np.searchsorted(self.cal, s, side="left")
        return float((1 + at_least) / (len(self.cal) + 1))

    def confidence(self, s: float) -> float:
        return 1.0 - self.p_value(s)
