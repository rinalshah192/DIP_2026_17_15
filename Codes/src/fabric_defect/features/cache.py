"""
Per-image patch-feature cache.
ECE501 - Digital Image Processing (Group 15)

Feature extraction dominates run time (LBP and GLCM on every window), while
model fitting and evaluation take milliseconds. Every block (GLCM, LBP, Gabor)
is therefore computed once per image and per window geometry, stored under
data/processed/, and reused by every run or ablation variant that only changes
modelling, scoring or thresholds.
"""

import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fabric_defect.core.config import PipelineConfig
from fabric_defect.features.descriptor import TextureDescriptor
from fabric_defect.features.preprocessing import Preprocessor, detect_fabric_roi, load_image, window_coords

log = logging.getLogger(__name__)

Scale = Tuple[int, int]  # (window size, stride)


@dataclass
class ImageFeatures:
    """Window features of one strip at one or more scales; coordinates are in full-strip space."""

    shape: Tuple[int, int]
    roi: Tuple[int, int]  # fabric column span (c0, c1)
    coords: Dict[int, np.ndarray]  # window size -> (N, 2) top-left (y, x)
    blocks: Dict[int, Dict[str, np.ndarray]]  # window size -> block name -> (N, D)


def compute_features(path: str, cfg: PipelineConfig, scales: Sequence[Scale]) -> ImageFeatures:
    """Load, crop to the fabric ROI, pre-process and describe every window of one strip."""
    img = load_image(path)
    c0, c1 = detect_fabric_roi(img, margin=cfg.preprocess.selvage_margin, min_width=max(s for s, _ in scales))
    norm = Preprocessor(cfg.preprocess).process(img[:, c0:c1])["normalized"]
    desc = TextureDescriptor(cfg.glcm, cfg.lbp, cfg.gabor)
    coords, blocks = {}, {}
    for size, stride in scales:
        local = window_coords(norm.shape, size, stride)
        blocks[size] = desc.extract_blocks(norm, local, size)
        coords[size] = np.array(local, dtype=np.int64).reshape(-1, 2) + np.array([0, c0])
    return ImageFeatures(shape=img.shape, roi=(c0, c1), coords=coords, blocks=blocks)


def _feature_key(cfg: PipelineConfig) -> str:
    """Hash of everything that changes the cached numbers (not modelling or scoring settings)."""
    relevant = {k: asdict(getattr(cfg, k)) for k in ("preprocess", "glcm", "lbp", "gabor")}
    return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


def _file_key(path: str) -> str:
    st = Path(path).stat()
    return hashlib.sha1(f"{Path(path).resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]


def _save(f: ImageFeatures, size: int, file: Path) -> None:
    """One file per image and scale, so runs with different window sets share entries."""
    arrays = {"shape": np.array(f.shape), "roi": np.array(f.roi), "coords": f.coords[size]}
    arrays.update({f"b_{name}": block for name, block in f.blocks[size].items()})
    tmp = file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(file)  # atomic: a crashed run never leaves a half-written cache entry


def _load(files: Dict[int, Path]) -> ImageFeatures:
    coords, blocks = {}, {}
    for size, file in files.items():
        with np.load(file) as z:
            shape, roi = tuple(int(v) for v in z["shape"]), tuple(int(v) for v in z["roi"])
            coords[size] = z["coords"]
            blocks[size] = {k[2:]: z[k] for k in z.files if k.startswith("b_")}
    return ImageFeatures(shape=shape, roi=roi, coords=coords, blocks=blocks)


def _load_or_compute(args) -> ImageFeatures:
    path, cfg, scales, files = args
    if files is not None and all(f.exists() for f in files.values()):
        try:
            return _load(files)
        except (OSError, ValueError, KeyError):
            log.warning("Recomputing unreadable cache entry for %s", path)
    f = compute_features(path, cfg, scales)
    if files is not None:
        for size, file in files.items():
            _save(f, size, file)
    return f


def load_features(
    paths: List[str],
    cfg: PipelineConfig,
    scales: Sequence[Scale],
    cache_dir: Optional[Path] = None,
    workers: int = 1
) -> Dict[str, ImageFeatures]:
    """Features for every path, from the cache when possible; `workers` > 1 uses processes."""
    folder = None
    if cache_dir is not None:
        folder = Path(cache_dir) / f"features-{_feature_key(cfg)}"
        folder.mkdir(parents=True, exist_ok=True)

    def files(p):
        return {size: folder / f"{_file_key(p)}_{size}s{stride}.npz" for size, stride in scales} if folder else None

    jobs = [(p, cfg, list(scales), files(p)) for p in paths]
    hits = sum(1 for *_, fs in jobs if fs is not None and all(f.exists() for f in fs.values()))
    log.info("Features for %d images (%d cached, %d to compute, %d worker%s)",
             len(jobs), hits, len(jobs) - hits, workers, "s" if workers > 1 else "")
    if workers > 1 and len(jobs) - hits > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_load_or_compute, jobs, chunksize=1))
    else:
        results = [_load_or_compute(j) for j in jobs]
    return dict(zip(paths, results))
