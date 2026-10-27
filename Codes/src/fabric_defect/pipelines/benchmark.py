"""
End-to-end benchmark run.
ECE501 - Digital Image Processing (Group 15)

catalog -> cached window features -> per-fabric reference models (fit set) ->
thresholds (held-out calibration set) -> score, segment and evaluate every
defective strip -> CSVs, run record and figures.
"""

import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fabric_defect.core.config import PipelineConfig
from fabric_defect.core.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR
from fabric_defect.data.afid import AITEXDataset, load_mask
from fabric_defect.data.synthetic import generate_synthetic_aitex_benchmark
from fabric_defect.evaluation.metrics import EvaluationSuite, evaluate_patch_classification, evaluate_pixel_localization
from fabric_defect.features.cache import ImageFeatures, load_features
from fabric_defect.features.preprocessing import Preprocessor, load_image
from fabric_defect.modeling.reference import TextureReferenceModel
from fabric_defect.modeling.segmentation import DefectSegmenter, deviation_map
from fabric_defect.visualization.plots import save_inspection_figure, save_summary_charts

log = logging.getLogger(__name__)

N_FIGURES = 3  # inspection collages per run


def _git_revision() -> dict:
    """Commit SHA and dirty flag of the checkout, or None outside a git repo."""
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True,
                                  text=True, check=True).stdout.strip()
        return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def write_run_info(out_dir: Path, cfg: PipelineConfig, data_dir: Path, synthetic: bool, **extra) -> Path:
    """Record everything needed to reproduce a run next to its results."""
    info = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_revision(),
        "data_dir": str(data_dir),
        "synthetic": synthetic,
        "python": platform.python_version(),
        "packages": {pkg: version(pkg) for pkg in
                     ("numpy", "opencv-python-headless", "scikit-image", "pandas", "matplotlib")},
        "config": cfg.to_dict(),
        **extra,
    }
    path = Path(out_dir) / "run_config.json"
    path.write_text(json.dumps(info, indent=2, default=str))
    return path


def split_images(paths, calibration_fraction, rng):
    """Image-level split of defect-free images into (fit, calibration) lists.

    Splitting by image (not by patch) keeps overlapping patches of one strip on
    the same side. At least one image always stays in the fit set.
    """
    paths = list(paths)
    if calibration_fraction <= 0 or len(paths) < 2:
        return paths, []
    n_calib = min(len(paths) - 1, max(1, round(len(paths) * calibration_fraction)))
    order = rng.permutation(len(paths))
    calib = [paths[i] for i in order[:n_calib]]
    fit = [paths[i] for i in order[n_calib:]]
    return fit, calib


def load_manifest(data_dir: Path, synthetic: bool) -> pd.DataFrame:
    if synthetic:
        log.warning("SYNTHETIC smoke run: results are not AFID results")
        return generate_synthetic_aitex_benchmark(Path(data_dir), samples_per_type=2)
    return AITEXDataset(data_dir).scan_catalog()


def select_test(defect_images: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """All defective strips (max_samples 0), or up to max_samples stratified across fabrics."""
    if max_samples <= 0:
        return defect_images
    per_fabric = max(1, max_samples // max(1, defect_images["fabric_type"].nunique()))
    # groupby().head keeps the fabric_type column (groupby().apply drops it on pandas >= 3)
    return defect_images.groupby("fabric_type").head(per_fabric).head(max_samples)


def feature_scales(cfg: PipelineConfig) -> List[Tuple[int, int]]:
    return [(cfg.patch.size, cfg.patch.stride)]


@dataclass
class Split:
    fit: Dict[str, List[str]]  # fabric -> defect-free image paths used to fit
    calib: Dict[str, List[str]]  # fabric -> held-out defect-free paths used to set tau


def split_by_fabric(clean: pd.DataFrame, cfg: PipelineConfig) -> Split:
    """Seeded per-fabric fit/calibration split; identical for every variant of one seed."""
    rng = np.random.default_rng(cfg.seed)
    fit, calib = {}, {}
    for fabric in sorted(clean["fabric_type"].unique()):
        paths = clean.loc[clean["fabric_type"] == fabric, "image_path"].tolist()
        fit[fabric], calib[fabric] = split_images(paths, cfg.model.calibration_fraction, rng)
    return Split(fit=fit, calib=calib)


def stack(feats: Dict[str, ImageFeatures], paths: List[str], size: int, use) -> Optional[np.ndarray]:
    """Descriptor rows of all windows of the given images, or None if there are none."""
    rows = [np.hstack([feats[p].blocks[size][b] for b in use]) for p in paths]
    rows = [r for r in rows if len(r)]
    return np.vstack(rows) if rows else None


def fit_models(feats: Dict[str, ImageFeatures], split: Split, cfg: PipelineConfig) -> Dict[str, TextureReferenceModel]:
    """One reference model per fabric, fitted on the fit images and calibrated on the held-out ones."""
    size, use = cfg.patch.size, cfg.features.use
    models = {}
    for fabric, fit_paths in split.fit.items():
        X_fit = stack(feats, fit_paths, size, use)
        if X_fit is None:
            log.warning("%s: no usable defect-free patches, fabric skipped", fabric)
            continue
        X_cal = stack(feats, split.calib[fabric], size, use)
        model = TextureReferenceModel(fabric_type=fabric, cfg=cfg.model).fit(X_fit, X_cal)
        source = (f"{len(split.calib[fabric])} held-out images" if model.calibrated_out_of_sample
                  else "in-sample (too few images)")
        log.debug("%s: fit on %d images / %d patches; tau = %.3f from %s",
                  fabric, len(fit_paths), len(X_fit), model.adaptive_threshold, source)
        models[fabric] = model
    return models


@dataclass
class StripResult:
    """Everything evaluate() computed for one strip; kept only for figures."""

    row: dict
    heatmap: np.ndarray
    gt_mask: np.ndarray
    pred_mask: np.ndarray
    threshold: float


def evaluate(
    test: pd.DataFrame,
    feats: Dict[str, ImageFeatures],
    models: Dict[str, TextureReferenceModel],
    cfg: PipelineConfig,
    keep: int = 0
) -> Tuple[pd.DataFrame, List[StripResult]]:
    """Score, segment and measure every test strip; returns per-image metrics and `keep` strips for figures."""
    size, use = cfg.patch.size, cfg.features.use
    segmenter = DefectSegmenter(cfg.segment, method="adaptive", adaptive_k=cfg.model.k)
    suite, kept = EvaluationSuite(), []
    label = f"{'+'.join(use)} ({cfg.model.scoring})"

    for row in test.to_dict("records"):
        name, fabric = row["filename"], row["fabric_type"]
        model = models.get(fabric)
        if model is None:
            log.warning("Skipping %s: no reference model for %s", name, fabric)
            continue
        if not row["mask_paths"]:
            # An empty ground truth would silently score every detection as a false positive
            log.warning("Skipping %s: no ground-truth mask found", name)
            continue
        f = feats[row["image_path"]]
        if len(f.coords[size]) == 0:
            log.warning("Skipping %s: fabric ROI smaller than one patch", name)
            continue
        gt = load_mask(row["mask_paths"], f.shape)
        if not gt.any():
            log.warning("%s: ground-truth mask is empty; recall-type metrics will be NaN", name)

        X = np.hstack([f.blocks[size][b] for b in use])
        scores, tau = model.score_strip(X)
        heat = deviation_map(scores, f.coords[size], f.shape, size)
        pred = segmenter.segment(heat, reference_threshold=tau)["cleaned_mask"]

        labels = [int(np.count_nonzero(gt[y:y + size, x:x + size]) > cfg.patch.defect_min_pixels)
                  for y, x in f.coords[size]]
        suite.log_result(
            sample_id=name, fabric_type=fabric, defect_type=row["defect_type"],
            patch_metrics=evaluate_patch_classification(labels, (scores >= tau).astype(int)),
            pixel_metrics=evaluate_pixel_localization(pred, gt),
            descriptor_used=label,
        )
        if len(kept) < keep and gt.any():
            kept.append(StripResult(row, heat, gt, pred, tau))
    return suite.to_dataframe(), kept


def run_full_pipeline(
    cfg: PipelineConfig = PipelineConfig(),
    data_dir: Path = RAW_DATA_DIR,
    out_dir: Path = RESULTS_DIR,
    synthetic: bool = False,
    cache_dir: Optional[Path] = PROCESSED_DATA_DIR,
    workers: int = 1
) -> pd.DataFrame:
    manifest = load_manifest(data_dir, synthetic)
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saved %s", write_run_info(out_dir, cfg, Path(data_dir), synthetic))

    clean = manifest[~manifest["is_defective"]]
    test = select_test(manifest[manifest["is_defective"]], cfg.max_test_samples)
    log.info("Catalogued %d images: %d defect-free; testing %d defective strips",
             len(manifest), len(clean), len(test))

    split = split_by_fabric(clean, cfg)
    feats = load_features(clean["image_path"].tolist() + test["image_path"].tolist(),
                          cfg, feature_scales(cfg), cache_dir, workers)
    models = fit_models(feats, split, cfg)
    results, kept = evaluate(test, feats, models, cfg, keep=N_FIGURES)
    return write_results(results, kept, cfg, out_dir, synthetic)


def write_results(results: pd.DataFrame, kept: List[StripResult], cfg: PipelineConfig,
                  out_dir: Path, synthetic: bool) -> pd.DataFrame:
    figures_dir = Path(out_dir) / "figures"
    results.to_csv(out_dir / "metrics_summary.csv", index=False)
    if results.empty:
        log.error("No test strips were evaluated")
        return results

    suite = EvaluationSuite()
    suite.records = results.to_dict("records")
    suite.summary_table("fabric_type").to_csv(out_dir / "fabric_summary.csv", index=False)
    suite.summary_table("defect_type").to_csv(out_dir / "defect_summary.csv", index=False)
    log.info("Saved metrics to %s", out_dir)

    preprocessor = Preprocessor(cfg.preprocess)
    for s in kept:
        raw = load_image(s.row["image_path"])
        norm = preprocessor.process(raw)["normalized"]
        stem = Path(s.row["filename"]).stem
        save_inspection_figure(
            out_file=figures_dir / f"inspection_{stem}.png", sample_id=stem, raw_img=raw, norm_img=norm,
            heatmap=s.heatmap, gt_mask=s.gt_mask, pred_mask=s.pred_mask, threshold=s.threshold,
            defect_type=s.row["defect_type"], fabric_type=s.row["fabric_type"],
        )

    log.info(
        "Summary%s over %d strips (NaN-skipping means): patch acc %.2f%%, patch F1 %.2f%%, "
        "pixel IoU %.2f%%, pixel Dice %.2f%%",
        " (SYNTHETIC)" if synthetic else "", len(results),
        results["patch_accuracy"].mean() * 100, results["patch_f1"].mean() * 100,
        results["pixel_iou"].mean() * 100, results["pixel_dice"].mean() * 100
    )
    save_summary_charts(results, figures_dir / "benchmark_performance_charts.png")
    return results
