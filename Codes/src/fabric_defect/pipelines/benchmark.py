"""
End-to-end benchmark run.
ECE501 - Digital Image Processing (Group 15)

catalog -> per-fabric fit / calibration / test split of the defect-free strips ->
cached window features -> per-fabric reference models (fit set) -> thresholds and
image-level calibration (held-out set) -> score, segment and evaluate every
defective strip and every held-out defect-free strip -> CSVs, run record, figures.
"""

import json
import logging
import platform
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fabric_defect.core.config import PipelineConfig
from fabric_defect.core.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR
from fabric_defect.data.afid import AITEXDataset, load_mask
from fabric_defect.data.synthetic import generate_synthetic_aitex_benchmark
from fabric_defect.evaluation.metrics import (
    EvaluationSuite,
    evaluate_patch_classification,
    evaluate_pixel_localization,
    image_level_metrics,
    summarize,
)
from fabric_defect.features.cache import ImageFeatures, load_features
from fabric_defect.features.preprocessing import Preprocessor, load_image, window_sums
from fabric_defect.modeling.baselines import OneClassBaseline
from fabric_defect.modeling.confidence import ConformalConfidence, image_score
from fabric_defect.modeling.refinement import coarse_region, refined_mask
from fabric_defect.modeling.reference import TextureReferenceModel
from fabric_defect.modeling.segmentation import DefectSegmenter, deviation_map
from fabric_defect.visualization.plots import save_inspection_figure, save_score_distributions, save_summary_charts

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


def split_images(paths, calibration_fraction, rng, test_fraction=0.0):
    """Image-level split of defect-free images into (fit, calibration, test) lists.

    Splitting by image (not by patch) keeps overlapping patches of one strip on
    the same side. At least one image always stays in the fit set; a test image
    is only held out when at least three images exist.
    """
    paths = list(paths)
    n = len(paths)
    if n < 2:
        return paths, [], []
    n_cal = max(1, round(n * calibration_fraction)) if calibration_fraction > 0 else 0
    n_test = max(1, round(n * test_fraction)) if test_fraction > 0 and n >= 3 else 0
    n_test = min(n_test, max(0, n - 1 - n_cal))
    n_cal = min(n_cal, n - 1 - n_test)
    order = rng.permutation(n)
    calib = [paths[i] for i in order[:n_cal]]
    test = [paths[i] for i in order[n_cal:n_cal + n_test]]
    fit = [paths[i] for i in order[n_cal + n_test:]]
    return fit, calib, test


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
    scales = [(cfg.patch.size, cfg.patch.stride)]
    if cfg.refine.enabled:
        scales.extend(cfg.refine.scales())
    return scales


@dataclass
class Split:
    fit: Dict[str, List[str]]  # fabric -> defect-free image paths used to fit
    calib: Dict[str, List[str]]  # fabric -> held-out defect-free paths used to set tau
    test: Dict[str, List[str]] = field(default_factory=dict)  # fabric -> defect-free image-level negatives


def split_by_fabric(clean: pd.DataFrame, cfg: PipelineConfig) -> Split:
    """Seeded per-fabric fit/calibration/test split; identical for every variant of one seed."""
    rng = np.random.default_rng(cfg.seed)
    fit, calib, test = {}, {}, {}
    for fabric in sorted(clean["fabric_type"].unique()):
        paths = clean.loc[clean["fabric_type"] == fabric, "image_path"].tolist()
        fit[fabric], calib[fabric], test[fabric] = split_images(
            paths, cfg.model.calibration_fraction, rng, cfg.model.test_fraction)
    return Split(fit=fit, calib=calib, test=test)


def stack(feats: Dict[str, ImageFeatures], paths: List[str], size: int, use) -> Optional[np.ndarray]:
    """Descriptor rows of all windows of the given images, or None if there are none."""
    rows = [np.hstack([feats[p].blocks[size][b] for b in use]) for p in paths]
    rows = [r for r in rows if len(r)]
    return np.vstack(rows) if rows else None


def _vstack(*parts) -> Optional[np.ndarray]:
    parts = [p for p in parts if p is not None and len(p)]
    return np.vstack(parts) if parts else None


def make_model(fabric: str, cfg: PipelineConfig):
    """Mahalanobis reference (mahalanobis / relative / hybrid scoring) or a one-class baseline."""
    if cfg.model.scoring in ("ocsvm", "iforest"):
        return OneClassBaseline(fabric, cfg.model, seed=cfg.seed)
    return TextureReferenceModel(fabric_type=fabric, cfg=cfg.model)


Extra = Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]  # fabric -> (fit rows, calibration rows)


def fit_models(
    feats: Dict[str, ImageFeatures],
    split: Split,
    cfg: PipelineConfig,
    size: Optional[int] = None,
    extra: Optional[Extra] = None
) -> Dict[str, TextureReferenceModel]:
    """One model per fabric (or one shared by all fabrics when model.reference is "global"),
    fitted on the fit images (+ extra rows) and calibrated on the held-out ones (+ extra rows)."""
    size, use, extra = size or cfg.patch.size, cfg.features.use, extra or {}
    if cfg.model.reference == "global":
        # Fixed baseline: pool every fabric's fit and calibration images into one model and one tau
        pooled = Split(fit={"*": [p for v in split.fit.values() for p in v]},
                       calib={"*": [p for v in split.calib.values() for p in v]})
        shared = fit_models(feats, pooled, replace(cfg, model=replace(cfg.model, reference="per_fabric")), size)
        return {fabric: shared["*"] for fabric in split.fit} if "*" in shared else {}
    models = {}
    for fabric, fit_paths in split.fit.items():
        x_fit, x_cal = extra.get(fabric, (None, None))
        X_fit = _vstack(stack(feats, fit_paths, size, use), x_fit)
        if X_fit is None:
            log.warning("%s: no usable defect-free patches, fabric skipped", fabric)
            continue
        X_cal = _vstack(stack(feats, split.calib[fabric], size, use), x_cal)
        model = make_model(fabric, cfg).fit(X_fit, X_cal)
        log.debug("%s @%d: fit on %d windows; tau = %.3f (%s)", fabric, size, len(X_fit),
                  model.adaptive_threshold, "held-out" if model.calibrated_out_of_sample else "in-sample")
        models[fabric] = model
    return models


@dataclass
class Fold:
    """The models that score one group of test strips."""

    models: Dict[str, TextureReferenceModel]
    refiners: Dict[int, Dict[str, TextureReferenceModel]]  # refine window size -> fabric -> model
    confidence: Optional[ConformalConfidence]


def build_fold(feats, split: Split, cfg: PipelineConfig, extra: Optional[Dict[int, Extra]] = None) -> Fold:
    """Coarse models, refinement models (Mahalanobis, refine.k) and the image-level calibrator."""
    extra = extra or {}
    models = fit_models(feats, split, cfg, extra=extra.get(cfg.patch.size))
    refiners = {}
    if cfg.refine.enabled:
        fine = replace(cfg, model=replace(cfg.model, scoring="mahalanobis", k=cfg.refine.k))
        refiners = {s: fit_models(feats, split, fine, size=s, extra=extra.get(s)) for s, _ in cfg.refine.scales()}
    size, use = cfg.patch.size, cfg.features.use
    cal_scores = []
    for fabric, model in models.items():
        for p in split.calib.get(fabric, []):
            if len(feats[p].coords[size]):
                scores, tau = model.score_strip(np.hstack([feats[p].blocks[size][b] for b in use]))
                cal_scores.append(image_score(scores, tau))
    confidence = ConformalConfidence(cal_scores) if cal_scores else None
    return Fold(models, refiners, confidence)


def window_gt(rows: pd.DataFrame, feats: Dict[str, ImageFeatures], scales) -> Dict[str, Dict[int, np.ndarray]]:
    """Ground-truth pixel count of every window of every defective strip, per window size."""
    out = {}
    for r in rows.to_dict("records"):
        if not r["mask_paths"]:
            continue
        f = feats[r["image_path"]]
        gt = load_mask(r["mask_paths"], f.shape)
        out[r["image_path"]] = {s: window_sums(gt, f.coords[s], s) for s, _ in scales}
    return out


def background(feats, gt_counts, paths: List[str], size: int, use) -> Optional[np.ndarray]:
    """Descriptor rows of the windows of defective strips that contain no defect pixel."""
    rows = [np.hstack([feats[p].blocks[size][b] for b in use])[gt_counts[p][size] == 0]
            for p in paths if p in gt_counts]
    return _vstack(*rows)


def assign_folds(test: pd.DataFrame, negatives: pd.DataFrame, cfg: PipelineConfig) -> Dict[str, int]:
    """Per fabric, deal its defective strips (shuffled) and its negatives round-robin over
    min(folds, n_defective) folds. Fabrics with fewer than 2 defective strips get fold -1,
    meaning "scored by the defect-free-only reference" (there is no other strip to learn from)."""
    rng = np.random.default_rng(cfg.seed + 1)
    folds = {}
    for fabric in sorted(set(test["fabric_type"]) | set(negatives["fabric_type"])):
        pos = test.loc[test["fabric_type"] == fabric, "image_path"].tolist()
        neg = negatives.loc[negatives["fabric_type"] == fabric, "image_path"].tolist()
        k = min(cfg.model.folds, len(pos))
        if k < 2:
            folds.update({p: -1 for p in pos + neg})
            continue
        folds.update({pos[i]: j % k for j, i in enumerate(rng.permutation(len(pos)))})
        folds.update({p: j % k for j, p in enumerate(neg)})
    return folds


def run_protocol(
    test: pd.DataFrame,
    negatives: pd.DataFrame,
    split: Split,
    feats: Dict[str, ImageFeatures],
    cfg: PipelineConfig,
    keep: int = 0
) -> Tuple[pd.DataFrame, List["StripResult"]]:
    """Evaluate defective strips and defect-free negatives under cfg.model.reference.

    per_fabric / global: one set of models scores everything. in_domain: every strip
    is scored by models whose extra background windows come only from *other*
    defective strips of its fabric (cross-validation by strip, never by window).
    """
    rows = pd.concat([test, negatives], ignore_index=True)
    if cfg.model.reference != "in_domain":
        return evaluate(rows, feats, build_fold(feats, split, cfg), cfg, keep=keep)

    scales = [(cfg.patch.size, cfg.patch.stride)] + (list(cfg.refine.scales()) if cfg.refine.enabled else [])
    use = cfg.features.use
    gt_counts = window_gt(test, feats, scales)
    folds = assign_folds(test[test["mask_paths"].apply(len) > 0], negatives, cfg)
    rows["fold"] = rows["image_path"].map(folds).fillna(-1).astype(int)
    parts, kept = [], []
    for j in sorted(rows["fold"].unique()):
        extra: Dict[int, Extra] = {s: {} for s, _ in scales}
        if j >= 0:
            rng = np.random.default_rng(cfg.seed + 2 + int(j))
            for fabric in sorted(test["fabric_type"].unique()):
                train = [p for p in test.loc[test["fabric_type"] == fabric, "image_path"]
                         if folds.get(p, -1) >= 0 and folds[p] != j]
                if not train:
                    continue
                fit_p, cal_p, _ = split_images(train, cfg.model.calibration_fraction, rng)
                for s, _ in scales:
                    extra[s][fabric] = (background(feats, gt_counts, fit_p, s, use),
                                        background(feats, gt_counts, cal_p, s, use))
        res, k = evaluate(rows[rows["fold"] == j], feats, build_fold(feats, split, cfg, extra), cfg,
                          keep=max(0, keep - len(kept)))
        parts.append(res)
        kept.extend(k)
    return pd.concat(parts, ignore_index=True), kept


@dataclass
class StripResult:
    """Everything evaluate() computed for one strip; kept only for figures."""

    row: dict
    heatmap: np.ndarray
    gt_mask: np.ndarray
    pred_mask: np.ndarray
    threshold: float
    coarse_mask: Optional[np.ndarray] = None  # 256-level mask when refinement replaced it


def evaluate(
    rows: pd.DataFrame,
    feats: Dict[str, ImageFeatures],
    fold: Fold,
    cfg: PipelineConfig,
    keep: int = 0
) -> Tuple[pd.DataFrame, List[StripResult]]:
    """Score, segment and measure every strip (defective and defect-free); returns per-strip
    metrics and up to `keep` defective strips for figures."""
    size, use = cfg.patch.size, cfg.features.use
    segmenter = DefectSegmenter(cfg.segment, method="adaptive", adaptive_k=cfg.model.k)
    suite, kept = EvaluationSuite(), []
    label = f"{'+'.join(use)} ({cfg.model.scoring}, {cfg.model.reference})"

    for row in rows.to_dict("records"):
        name, fabric, defective = row["filename"], row["fabric_type"], bool(row["is_defective"])
        model = fold.models.get(fabric)
        if model is None:
            log.warning("Skipping %s: no reference model for %s", name, fabric)
            continue
        if defective and not row["mask_paths"]:
            # An empty ground truth would silently score every detection as a false positive
            log.warning("Skipping %s: no ground-truth mask found", name)
            continue
        f = feats[row["image_path"]]
        if len(f.coords[size]) == 0:
            log.warning("Skipping %s: fabric ROI smaller than one patch", name)
            continue
        gt = load_mask(row["mask_paths"], f.shape) if defective else np.zeros(f.shape, np.uint8)
        if defective and not gt.any():
            log.warning("%s: ground-truth mask is empty; recall-type metrics will be NaN", name)

        X = np.hstack([f.blocks[size][b] for b in use])
        scores, tau = model.score_strip(X)
        heat = deviation_map(scores, f.coords[size], f.shape, size)
        pred = segmenter.segment(heat, reference_threshold=tau)["cleaned_mask"]
        coarse = None
        if fold.refiners:
            coarse = pred
            region = coarse_region(scores >= tau, f.coords[size], f.shape, size)
            for s, _ in cfg.refine.scales():
                fine = fold.refiners[s].get(fabric)
                Xf = np.hstack([f.blocks[s][b] for b in use])
                fs = fine.score_batch(Xf) if fine is not None else np.zeros(len(Xf))
                tf = fine.adaptive_threshold if fine is not None else np.inf
                region = refined_mask(region, f.coords[s], fs, tf, s) > 0
            pred = segmenter.apply_morphological_cleanup(region.astype(np.uint8) * 255)

        s_img = image_score(scores, tau)
        conf = fold.confidence.confidence(s_img) if fold.confidence else float("nan")
        labels = (window_sums(gt, f.coords[size], size) > cfg.patch.defect_min_pixels).astype(int)
        suite.log_result(
            sample_id=name, fabric_type=fabric, defect_type=row["defect_type"],
            patch_metrics=evaluate_patch_classification(labels, (scores >= tau).astype(int)),
            pixel_metrics=evaluate_pixel_localization(pred, gt),
            descriptor_used=label,
            # Threshold-free: how well this strip's window scores rank defect windows above normal ones
            patch_auroc=float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else float("nan"),
            is_defective=defective, image_score=s_img, confidence=conf,
            image_pred=bool(np.isfinite(conf) and 1.0 - conf <= cfg.model.image_alpha),
            fold=int(row.get("fold", -1)) if pd.notna(row.get("fold", -1)) else -1,
        )
        if defective and len(kept) < keep and gt.any():
            kept.append(StripResult(row, heat, gt, pred, tau, coarse))
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
    split = split_by_fabric(clean, cfg)
    negatives = clean[clean["image_path"].isin([p for v in split.test.values() for p in v])]
    log.info("Catalogued %d images: %d defect-free (%d held out as negatives); testing %d defective strips",
             len(manifest), len(clean), len(negatives), len(test))

    feats = load_features(clean["image_path"].tolist() + test["image_path"].tolist(),
                          cfg, feature_scales(cfg), cache_dir, workers)
    results, kept = run_protocol(test, negatives, split, feats, cfg, keep=N_FIGURES)
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
    (Path(out_dir) / "summary.json").write_text(json.dumps(summarize(results), indent=2))
    per_fabric = [{"fabric_type": fab, **image_level_metrics(g)} for fab, g in results.groupby("fabric_type")]
    pd.DataFrame(per_fabric).to_csv(out_dir / "image_level_by_fabric.csv", index=False)
    log.info("Saved metrics to %s", out_dir)

    preprocessor = Preprocessor(cfg.preprocess)
    for s in kept:
        raw = load_image(s.row["image_path"])
        norm = preprocessor.process(raw)["normalized"]
        stem = Path(s.row["filename"]).stem
        save_inspection_figure(
            out_file=figures_dir / f"inspection_{stem}.png", sample_id=stem, raw_img=raw, norm_img=norm,
            heatmap=s.heatmap, gt_mask=s.gt_mask, pred_mask=s.pred_mask, threshold=s.threshold,
            defect_type=s.row["defect_type"], fabric_type=s.row["fabric_type"], coarse_mask=s.coarse_mask,
        )

    s = summarize(results)
    log.info(
        "Summary%s over %d defective strips: patch F1 %.2f%% (pooled %.2f%%), pixel IoU %.2f%% (pooled %.2f%%), "
        "pixel recall %.2f%%; image level (%d+/%d-): accuracy %.2f%%, AUROC %.3f",
        " (SYNTHETIC)" if synthetic else "", s["n_images"], 100 * s["patch_f1"], 100 * s["pooled_patch_f1"],
        100 * s["pixel_iou"], 100 * s["pooled_pixel_iou"], 100 * s["pixel_recall"],
        s.get("image_n_pos", 0), s.get("image_n_neg", 0), 100 * s.get("image_accuracy", float("nan")),
        s.get("image_auroc", float("nan"))
    )
    defective = results[results["is_defective"].astype(bool)]
    save_summary_charts(defective, figures_dir / "benchmark_performance_charts.png")
    save_score_distributions(results, figures_dir / "image_scores.png")
    return results
