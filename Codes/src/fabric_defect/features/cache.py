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


def _feature_key(cfg: PipelineConfig, scales: Sequence[Scale]) -> str:
    """Hash of everything that changes the cached numbers (not modelling or scoring settings)."""
    relevant = {k: asdict(getattr(cfg, k)) for k in ("preprocess", "glcm", "lbp", "gabor")}
    relevant["scales"] = [list(s) for s in scales]
    return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


def _file_key(path: str) -> str:
    st = Path(path).stat()
    return hashlib.sha1(f"{Path(path).resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]


def _save(f: ImageFeatures, file: Path) -> None:
    arrays = {"shape": np.array(f.shape), "roi": np.array(f.roi)}
    for size in f.coords:
        arrays[f"{size}_coords"] = f.coords[size]
        for name, block in f.blocks[size].items():
            arrays[f"{size}_{name}"] = block
    tmp = file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(file)  # atomic: a crashed run never leaves a half-written cache entry


def _load(file: Path) -> ImageFeatures:
    with np.load(file) as z:
        sizes = sorted({int(k.split("_")[0]) for k in z.files if k.endswith("_coords")})
        return ImageFeatures(
            shape=tuple(int(v) for v in z["shape"]),
            roi=tuple(int(v) for v in z["roi"]),
            coords={s: z[f"{s}_coords"] for s in sizes},
            blocks={s: {k.split("_", 1)[1]: z[k] for k in z.files
                        if k.startswith(f"{s}_") and not k.endswith("_coords")} for s in sizes},
        )


def _load_or_compute(args) -> ImageFeatures:
    path, cfg, scales, cache_file = args
    if cache_file is not None and cache_file.exists():
        try:
            return _load(cache_file)
        except (OSError, ValueError, KeyError):
            log.warning("Recomputing unreadable cache entry %s", cache_file)
    f = compute_features(path, cfg, scales)
    if cache_file is not None:
        _save(f, cache_file)
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
        folder = Path(cache_dir) / f"features-{_feature_key(cfg, scales)}"
        folder.mkdir(parents=True, exist_ok=True)
    jobs = [(p, cfg, list(scales), folder / f"{_file_key(p)}.npz" if folder else None) for p in paths]
    hits = sum(1 for *_, c in jobs if c is not None and c.exists())
    log.info("Features for %d images (%d cached, %d to compute, %d worker%s)",
             len(jobs), hits, len(jobs) - hits, workers, "s" if workers > 1 else "")
    if workers > 1 and len(jobs) - hits > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_load_or_compute, jobs, chunksize=1))
    else:
        results = [_load_or_compute(j) for j in jobs]
    return dict(zip(paths, results))
