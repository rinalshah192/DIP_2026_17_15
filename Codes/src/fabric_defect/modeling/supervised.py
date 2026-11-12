"""
Supervised (mask-trained) detector.
ECE501 - Digital Image Processing (Group 15)

The one-class references learn only what normal fabric looks like. AFID also
ships a pixel mask for every defective strip, so a classifier can learn what
defects look like as well. Both stages use classical features and gradient-
boosted trees (Friedman, 2001); only the decision function is learned.

1. Window stage: every 32x32 window (stride 16) is described by its GLCM, LBP
   and Gabor blocks, the same blocks robust-z-scored within its strip (which
   cancels the capture shift between AFID's defect-free and defective images),
   the strip-normalised blocks of the 64x64 and 256x256 windows around it
   (context), and a one-hot fabric code. Label: window holds any defect pixel.
2. Pixel stage: inside windows the first stage flags, each pixel is classified
   from dense filter-bank responses (features/pixel.py).

All training and threshold choice happen on strips that are never scored with
the result (see pipelines/supervised.py).
"""

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from fabric_defect.core.config import SupervisedConfig
from fabric_defect.features.cache import ImageFeatures

N_FABRICS = 7
BLOCKS = ("glcm", "lbp", "gabor")


def _block(f: ImageFeatures, size: int) -> np.ndarray:
    return np.hstack([f.blocks[size][b] for b in BLOCKS]).astype(np.float32)


def _strip_z(a: np.ndarray) -> np.ndarray:
    """Robust z-score of every column within one strip."""
    med = np.median(a, axis=0)
    return (a - med) / (np.median(np.abs(a - med), axis=0) + 1e-6)


def nearest_on_grid(fine: np.ndarray, fine_size: int, grid: np.ndarray, grid_size: int) -> np.ndarray:
    """Index of the grid window whose centre is nearest each fine window's centre.

    Window coordinates come from window_coords, i.e. a row-major product of y and x starts,
    so the nearest centre is found per axis with a binary search: O(n log m) time and O(n)
    memory, where a full distance matrix would be 15k x 3.6k for 16x16 windows.
    """
    ys, xs = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    if len(ys) * len(xs) != len(grid):
        raise ValueError("context windows do not form a grid")

    def nearest(starts, q):
        c = starts + grid_size / 2
        i = np.clip(np.searchsorted(c, q), 1, len(c) - 1) if len(c) > 1 else np.zeros(len(q), int)
        if len(c) > 1:
            i = np.where(np.abs(c[i - 1] - q) <= np.abs(c[i] - q), i - 1, i)
        return i

    centre = fine + fine_size / 2
    return nearest(ys, centre[:, 0]) * len(xs) + nearest(xs, centre[:, 1])


def window_matrix(f: ImageFeatures, fabric: str, cfg: SupervisedConfig, window: Optional[int] = None) -> np.ndarray:
    """Feature rows of every window of one strip at size `window` (default cfg.window); context comes
    from every cfg.context size larger than it (see module docstring)."""
    s = window or cfg.window
    raw = _block(f, s)
    parts = [raw, _strip_z(raw)]
    for c in (c for c in cfg.context if c > s):
        z = _strip_z(_block(f, c))
        parts.append(z[nearest_on_grid(f.coords[s], s, f.coords[c], c)])
    onehot = np.zeros((len(raw), N_FABRICS), np.float32)
    code = int(fabric[-2:]) if fabric[-2:].isdigit() else -1
    if 0 <= code < N_FABRICS:
        onehot[:, code] = 1
    parts.append(onehot)
    return np.hstack(parts)


def _booster(cfg: SupervisedConfig, seed: int, leaves: Optional[int] = None) -> HistGradientBoostingClassifier:
    # Early stopping holds out 10 % of the *training rows* (never test strips) and stops after
    # n_iter_no_change rounds without improvement, so max_iter is only an upper bound
    return HistGradientBoostingClassifier(
        max_iter=cfg.max_iter, learning_rate=cfg.learning_rate, max_leaf_nodes=leaves or cfg.max_leaf_nodes,
        l2_regularization=1.0, early_stopping=cfg.early_stopping, n_iter_no_change=cfg.n_iter_no_change,
        validation_fraction=0.1, random_state=seed)


class WindowClassifier:
    """Gradient-boosted window classifier; negatives are subsampled, positives kept."""

    def __init__(self, cfg: SupervisedConfig = SupervisedConfig(), seed: int = 0):
        self.cfg, self.seed = cfg, seed
        self.model = _booster(cfg, seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WindowClassifier":
        """Fit on (already sampled) rows; positives are weighted so both classes count equally."""
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos == 0:
            raise ValueError("WindowClassifier needs at least one defect window to train on.")
        self.model.fit(X, y, sample_weight=np.where(y == 1, max(1.0, n_neg / n_pos), 1.0))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1].astype(np.float32)


class PixelClassifier:
    """Gradient-boosted per-pixel classifier on dense filter-bank features."""

    def __init__(self, cfg: SupervisedConfig = SupervisedConfig(), seed: int = 0):
        self.model = _booster(cfg, seed, leaves=max(cfg.max_leaf_nodes, 63))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PixelClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        """Probability map (H, W) for a (H, W, F) feature stack."""
        h, w, d = F.shape
        return self.model.predict_proba(F.reshape(-1, d))[:, 1].reshape(h, w).astype(np.float32)


def sample_pixels(F: np.ndarray, gt: np.ndarray, n: int, rng: np.random.Generator) -> tuple:
    """Training pixels of one strip: up to n defect pixels, n defect-free pixels within 15 px of a
    defect (the hard cases at its border) and n random pixels."""
    import cv2

    flat, lab = F.reshape(-1, F.shape[-1]), gt.ravel().astype(bool)
    pos = np.flatnonzero(lab)
    if len(pos) > n:
        pos = rng.choice(pos, n, replace=False)
    near = np.flatnonzero(cv2.dilate(gt.astype(np.uint8), np.ones((31, 31), np.uint8)).ravel().astype(bool) & ~lab)
    near = rng.choice(near, min(len(near), n), replace=False) if len(near) else near
    rnd = rng.choice(len(lab), min(len(lab), n), replace=False)
    idx = np.concatenate([pos, near, rnd])
    return flat[idx].copy(), lab[idx].astype(int)


def sample_windows(X: np.ndarray, y: np.ndarray, cfg: SupervisedConfig, rng: np.random.Generator):
    """Training rows of one strip: every defect window, negatives_per_positive defect-free windows per
    defect window, and at least min_negatives defect-free windows (so defect-free strips count too)."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_neg = min(len(neg), max(cfg.min_negatives, cfg.negatives_per_positive * len(pos)))
    idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]) if n_neg else pos
    return X[idx].copy(), y[idx].copy()


def stack_windows(feats: Dict[str, ImageFeatures], rows: Sequence[dict], cfg: SupervisedConfig,
                  labels: Dict[str, np.ndarray]):
    """(X, y, strip index) over the given strips; labels map image path -> window defect-pixel counts."""
    X, y, g = [], [], []
    for i, r in enumerate(rows):
        f = feats[r["image_path"]]
        X.append(window_matrix(f, r["fabric_type"], cfg))
        c = labels.get(r["image_path"])
        y.append((c > 0).astype(int) if c is not None else np.zeros(len(f.coords[cfg.window]), int))
        g.append(np.full(len(f.coords[cfg.window]), i))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def hough_link(mask: np.ndarray, region: np.ndarray, min_length: int, max_gap: int = 16,
               thickness: int = 3, max_tilt_deg: float = 10.0) -> np.ndarray:
    """Join dashed thin detections into whole yarn lines (probabilistic Hough, Matas et al., 2000).

    Only segments within max_tilt_deg of horizontal (weft) or vertical (warp) are kept: those are
    the only directions AFID line defects (broken picks, broken ends, weft cracks) run in. Drawn
    segments are clipped to `region` (windows the window stage flagged), so the weave's own
    straight yarns elsewhere cannot add pixels.
    """
    import cv2

    lines = cv2.HoughLinesP((mask > 0).astype(np.uint8) * 255, 1, np.pi / 180,
                            threshold=max(10, min_length // 2), minLineLength=min_length, maxLineGap=max_gap)
    out = (mask > 0).astype(np.uint8) * 255
    if lines is None:
        return out
    drawn = np.zeros_like(out)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):  # OpenCV 4 returns (N, 1, 4), OpenCV 5 (N, 4)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        if min(angle, 180 - angle) <= max_tilt_deg or abs(angle - 90) <= max_tilt_deg:
            cv2.line(drawn, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness)
    return np.where((drawn > 0) & region, 255, out).astype(np.uint8)
