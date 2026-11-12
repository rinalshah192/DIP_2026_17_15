"""
Supervised benchmark with strip-grouped cross-validation.
ECE501 - Digital Image Processing (Group 15)

Every strip (defective with a mask, and defect-free) is scored exactly once, by
models that never saw it:

  outer fold k:  test strips E_k  |  training strips T_k = fit F_k + validation V_k
    1. train window classifiers (16x16 and 32x32) and the pixel classifier on F_k
    2. choose everything else on V_k by F1: which window map (16, 32 or their mean),
       window-map vs pixel-stage mask, gate / pixel thresholds, Hough linking, and the
       16x16 / 32x32 window, 256x256 patch and image decision thresholds
    3. retrain on all of T_k with the same settings, score E_k with the V_k choices

Folds are grouped by strip and stratified by (defective, fabric), so overlapping
windows of one strip never sit on both sides and every fold sees every fabric.
"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from fabric_defect.core.config import PipelineConfig
from fabric_defect.core.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR
from fabric_defect.data.afid import load_mask
from fabric_defect.evaluation.metrics import (
    EvaluationSuite,
    evaluate_patch_classification,
    evaluate_pixel_localization,
)
from fabric_defect.features.cache import ImageFeatures, load_features
from fabric_defect.features.pixel import PixelFeatureExtractor
from fabric_defect.features.preprocessing import Preprocessor, load_image, window_sums
from fabric_defect.modeling.segmentation import DefectSegmenter, deviation_map
from fabric_defect.modeling.supervised import (
    PixelClassifier,
    WindowClassifier,
    hough_link,
    sample_pixels,
    sample_windows,
    window_matrix,
)
from fabric_defect.pipelines.benchmark import N_FIGURES, StripResult, load_manifest, write_results, write_run_info

log = logging.getLogger(__name__)

GRID = np.round(np.arange(0.1, 0.96, 0.05), 2)  # candidate probability thresholds
GATES = np.round(np.arange(0.1, 0.91, 0.1), 2)  # coarser grid for the pixel-stage gate
HOUGH = [None, (32, 3), (64, 3), (128, 3), (64, 5)]  # (min segment length, line thickness) or off


def scales(cfg: PipelineConfig) -> List[Tuple[int, int]]:
    """Window geometries the supervised detector reads from the feature cache."""
    sizes = sorted(set(cfg.supervised.levels()) | set(cfg.supervised.context))
    return [(s, cfg.patch.stride if s == cfg.patch.size else s // 2) for s in sizes]


@dataclass
class Strip:
    row: dict
    feats: ImageFeatures
    counts: Dict[int, np.ndarray]  # classified window size -> defect pixels per window
    patch_counts: np.ndarray  # defect pixels per 256x256 patch
    gt: Optional[np.ndarray]  # full-strip mask (defective strips)

    def X(self, level: int, cfg: PipelineConfig) -> np.ndarray:
        """Window feature rows at one level, built on demand (16x16 rows of all strips would not fit)."""
        return window_matrix(self.feats, self.row["fabric_type"], cfg.supervised, level)


@dataclass
class Thresholds:
    gate: float  # window-map value a pixel must reach (pixel stage)
    pixel: float  # pixel probability (pixel stage) or window-map cut-off (no pixel stage)
    patch: float  # 256-patch decision on the max of the chosen map inside the patch
    image: float  # strip decision on the max of the chosen map
    hough: Optional[Tuple[int, int]] = None  # Hough linking (min length, thickness), or off
    use_pixel: bool = True  # pixel-stage mask; False: window map >= `pixel`
    map: str = "16"  # window map used for the mask, patches and image: a level ("16", "32") or "mean"
    window: Dict[int, float] = field(default_factory=dict)  # decision per classified window, per level


class Samples:
    """Training rows drawn once per strip (and level), so every fold reuses the same draw."""

    def __init__(self, strips: List[Strip], cfg: PipelineConfig):
        self.strips, self.cfg = strips, cfg
        self.windows: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self.pixels: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.pre, self.px = Preprocessor(cfg.preprocess), PixelFeatureExtractor()

    def window(self, i: int, level: int):
        if (i, level) not in self.windows:
            s = self.strips[i]
            rng = np.random.default_rng([self.cfg.seed, i, level])
            self.windows[(i, level)] = sample_windows(s.X(level, self.cfg), (s.counts[level] > 0).astype(int),
                                                      self.cfg.supervised, rng)
        return self.windows[(i, level)]

    def pixel_features(self, s: Strip) -> np.ndarray:
        c0, c1 = s.feats.roi
        norm = self.pre.process(load_image(s.row["image_path"])[:, c0:c1])["normalized"]
        return self.px.extract(norm)

    def pixel(self, i: int):
        if i not in self.pixels:
            s = self.strips[i]
            c0, c1 = s.feats.roi
            gt = s.gt[:, c0:c1] > 0 if s.gt is not None else np.zeros((s.feats.shape[0], c1 - c0), bool)
            rng = np.random.default_rng([self.cfg.seed, i, 1])
            self.pixels[i] = sample_pixels(self.pixel_features(s), gt, self.cfg.supervised.pixel_samples, rng)
        return self.pixels[i]


def build_strips(rows: pd.DataFrame, feats: Dict[str, ImageFeatures], cfg: PipelineConfig) -> List[Strip]:
    p = cfg.patch.size
    out = []
    for r in rows.to_dict("records"):
        f = feats[r["image_path"]]
        gt = load_mask(r["mask_paths"], f.shape) if r["is_defective"] else None
        count = (lambda size: window_sums(gt, f.coords[size], size) if gt is not None  # noqa: E731
                 else np.zeros(len(f.coords[size]), int))
        out.append(Strip(row=r, feats=f, counts={L: count(L) for L in cfg.supervised.levels()},
                         patch_counts=count(p), gt=gt))
    return out


@dataclass
class Models:
    windows: Dict[int, WindowClassifier]
    pixel: Optional[PixelClassifier]


def train(samples: Samples, idx, cfg: PipelineConfig) -> Models:
    windows = {}
    for L in cfg.supervised.levels():
        parts = [samples.window(i, L) for i in idx]
        windows[L] = WindowClassifier(cfg.supervised, seed=cfg.seed).fit(
            np.vstack([a for a, _ in parts]), np.concatenate([b for _, b in parts]))
    pixel = None
    if cfg.supervised.pixel_stage:
        parts = [samples.pixel(i) for i in idx]
        pixel = PixelClassifier(cfg.supervised, seed=cfg.seed).fit(
            np.vstack([a for a, _ in parts]), np.concatenate([b for _, b in parts]))
    return Models(windows, pixel)


@dataclass
class Scored:
    """Model outputs for one strip; maps and pixels only for defective strips (memory)."""

    probs: Dict[int, np.ndarray]  # level -> per-window probability
    image: Dict[str, float]  # map name -> max value (image score)
    maps: Dict[str, np.ndarray] = field(default_factory=dict)  # map name -> full-strip probability map
    pix_prob: Optional[np.ndarray] = None  # pixel probability map (full strip)


def score(s: Strip, models: Models, samples: Samples, cfg: PipelineConfig) -> Scored:
    probs, maps = {}, {}
    for L, clf in models.windows.items():
        probs[L] = clf.predict(s.X(L, cfg))
        maps[str(L)] = deviation_map(probs[L], s.feats.coords[L], s.feats.shape, L)
    if len(maps) > 1:
        maps["mean"] = np.mean(list(maps.values()), axis=0)
    image = {k: float(v.max()) for k, v in maps.items()}
    if s.gt is None:
        return Scored(probs, image)
    pix = None
    if models.pixel is not None:
        c0, c1 = s.feats.roi
        pix = np.zeros(s.feats.shape, np.float32)
        # pixels below the lowest gate on every map can never reach the mask: skip them
        cand = np.max([m[:, c0:c1] for m in maps.values()], axis=0) >= GATES[0]
        if cand.any():
            pix[:, c0:c1][cand] = models.pixel.model.predict_proba(samples.pixel_features(s)[cand])[:, 1]
    return Scored(probs, image, {k: v.astype(np.float16) for k, v in maps.items()}, pix)


def mask(sc: Scored, th: Thresholds, segmenter: DefectSegmenter) -> np.ndarray:
    heat = sc.maps[th.map]
    if th.use_pixel and sc.pix_prob is not None:
        raw = (sc.pix_prob >= th.pixel) & (heat >= th.gate)
    else:
        raw = heat >= th.pixel
    raw = raw.astype(np.uint8) * 255
    if th.hough is not None:
        raw = hough_link(raw, heat >= th.gate, min_length=th.hough[0], thickness=th.hough[1])
    # keep thin defects: drop only isolated specks, no opening that would erase 1-2 px lines
    return segmenter.drop_small_components(raw)


def _pooled_f1(strips: List[Strip], scored: Dict[int, Scored], idx, th: Thresholds,
               segmenter: DefectSegmenter) -> float:
    """Pooled pixel F1 over the given defective strips (pooled IoU = F1 / (2 - F1), same ranking)."""
    tp = fp = fn = 0
    for i in idx:
        pred, gt = mask(scored[i], th, segmenter) > 0, strips[i].gt > 0
        tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def patch_scores(s: Strip, sc: Scored, name: str, cfg: PipelineConfig) -> np.ndarray:
    """Max of a window map inside every 256x256 patch."""
    p, heat = cfg.patch.size, sc.maps[name]
    return np.array([float(heat[y:y + p, x:x + p].max()) for y, x in s.feats.coords[p]])


def _best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    f1 = [2 * ((scores >= t) & (labels == 1)).sum() / max(1, (scores >= t).sum() + labels.sum()) for t in GRID]
    return float(GRID[int(np.argmax(f1))])


def choose_thresholds(strips: List[Strip], scored: Dict[int, Scored], cfg: PipelineConfig,
                      segmenter: DefectSegmenter) -> Thresholds:
    """Every choice from validation strips only, each maximising F1 (see module docstring)."""
    defective = [i for i in scored if strips[i].gt is not None]
    names = list(scored[defective[0]].maps)
    # 1. window-map masks for every map; the best one also gates the pixel stage
    window_only = {(n, t): _pooled_f1(strips, scored, defective, Thresholds(0.0, t, 0.5, 0.5, None, False, n), segmenter)
                   for n in names for t in GRID}
    (name, t_win), best_f1 = max(window_only.items(), key=lambda kv: kv[1])
    best = Thresholds(0.0, t_win, 0.5, 0.5, None, False, name)
    # 2. pixel stage on that map: only kept if it beats the window-map mask
    if any(scored[i].pix_prob is not None for i in defective):
        for g in GATES:
            for t in GRID:
                th = Thresholds(float(g), float(t), 0.5, 0.5, None, True, name)
                f1 = _pooled_f1(strips, scored, defective, th, segmenter)
                if f1 > best_f1:
                    best, best_f1 = th, f1
    # 3. Hough linking (ties keep it off)
    if cfg.supervised.hough_linking:
        options = {h: _pooled_f1(strips, scored, defective, Thresholds(**{**asdict(best), "hough": h}), segmenter)
                   for h in HOUGH}
        best.hough = max(options, key=lambda h: (options[h], h is None))
    # 4. decision thresholds
    best.patch = _best_f1(np.concatenate([patch_scores(strips[i], scored[i], name, cfg) for i in defective]),
                          np.concatenate([(strips[i].patch_counts > cfg.patch.defect_min_pixels) for i in defective]))
    best.window = {L: _best_f1(np.concatenate([scored[i].probs[L] for i in defective]),
                               np.concatenate([(strips[i].counts[L] > 0) for i in defective]))
                   for L in cfg.supervised.levels()}
    best.image = _best_f1(np.array([scored[i].image[name] for i in scored]),
                          np.array([strips[i].gt is not None for i in scored]))
    return best


def _fit_split(train: np.ndarray, strata: np.ndarray, cfg: PipelineConfig, fold: int):
    """Split one outer fold's training strips into fit and validation strips: per (defective,
    fabric) stratum, round(validation_fraction * n) strips go to validation, but a stratum with a
    single strip keeps it for fitting."""
    rng = np.random.default_rng(cfg.seed + 100 + fold)
    val = []
    for st in np.unique(strata[train]):
        members = train[strata[train] == st]
        n_val = min(len(members) - 1, round(len(members) * cfg.supervised.validation_fraction))
        val.extend(rng.permutation(members)[:max(0, n_val)])
    val = np.sort(np.array(val, dtype=int))
    return np.setdiff1d(train, val), val


def run_supervised(
    cfg: PipelineConfig = PipelineConfig(),
    data_dir: Path = RAW_DATA_DIR,
    out_dir: Path = RESULTS_DIR,
    synthetic: bool = False,
    cache_dir: Optional[Path] = PROCESSED_DATA_DIR,
    workers: int = 1
) -> pd.DataFrame:
    manifest = load_manifest(data_dir, synthetic)
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    log.info("Saved %s", write_run_info(out_dir, cfg, Path(data_dir), synthetic))

    usable = manifest[(~manifest["is_defective"]) | (manifest["mask_paths"].apply(len) > 0)].reset_index(drop=True)
    feats = load_features(usable["image_path"].tolist(), cfg, scales(cfg), cache_dir, workers)
    strips = build_strips(usable, feats, cfg)
    log.info("Supervised CV over %d strips (%d defective, %d defect-free); window levels %s",
             len(strips), int(usable["is_defective"].sum()), int((~usable["is_defective"]).sum()),
             cfg.supervised.levels())

    # (defective, fabric) strata; str() keeps them deterministic (hash() is salted per process)
    strata = pd.factorize(usable["is_defective"].astype(str) + "_" + usable["fabric_type"], sort=True)[0]
    outer = StratifiedGroupKFold(n_splits=cfg.supervised.folds, shuffle=True, random_state=cfg.seed)
    segmenter = DefectSegmenter(cfg.segment)
    samples = Samples(strips, cfg)
    suite, kept, chosen = EvaluationSuite(), [], []

    for k, (train_i, test_i) in enumerate(outer.split(np.zeros(len(strips)), strata, groups=np.arange(len(strips)))):
        fit, val = _fit_split(train_i, strata, cfg, k)
        models = train(samples, fit, cfg)
        th = choose_thresholds(strips, {i: score(strips[i], models, samples, cfg) for i in val}, cfg, segmenter)
        chosen.append({"fold": k, **asdict(th)})
        log.info("Fold %d: %d train / %d test strips; chose %s", k, len(train_i), len(test_i), th)

        models = train(samples, train_i, cfg)
        for i in test_i:
            log_strip(suite, strips[i], score(strips[i], models, samples, cfg), th, cfg, segmenter, k, kept)

    results = suite.to_dataframe()
    pd.DataFrame(chosen).to_csv(out_dir / "thresholds.csv", index=False)
    return write_results(results, kept, cfg, out_dir, synthetic)


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else float("nan")


def log_strip(suite: EvaluationSuite, s: Strip, sc: Scored, th: Thresholds, cfg: PipelineConfig,
              segmenter: DefectSegmenter, fold: int, kept: List[StripResult]) -> None:
    r, defective = s.row, s.gt is not None
    gt = s.gt if defective else np.zeros(s.feats.shape, np.uint8)
    pred = mask(sc, th, segmenter) if defective else np.zeros(s.feats.shape, np.uint8)
    labels = (s.patch_counts > cfg.patch.defect_min_pixels).astype(int)
    patch = patch_scores(s, sc, th.map, cfg) if defective else np.zeros(len(labels))
    per_level = {}
    for L in cfg.supervised.levels():
        lab = (s.counts[L] > 0).astype(int)
        m = evaluate_patch_classification(lab, (sc.probs[L] >= th.window[L]).astype(int))
        per_level.update({f"window{L}_{k[6:]}": v for k, v in m.items()})
        per_level[f"window{L}_auroc"] = _auroc(lab, sc.probs[L])
    img = sc.image[th.map]
    suite.log_result(
        sample_id=r["filename"], fabric_type=r["fabric_type"], defect_type=r["defect_type"],
        patch_metrics=evaluate_patch_classification(labels, (patch >= th.patch).astype(int)),
        pixel_metrics=evaluate_pixel_localization(pred, gt),
        descriptor_used=f"supervised ({'pixel stage' if th.use_pixel and sc.pix_prob is not None else 'window map'}"
                        f", map {th.map})",
        patch_auroc=_auroc(labels, patch), is_defective=defective, image_score=img, confidence=img,
        image_pred=bool(img >= th.image), fold=fold, **per_level,
    )
    if defective and len(kept) < N_FIGURES and gt.any():
        heat = sc.maps[th.map].astype(np.float32)
        coarse = ((heat >= th.gate).astype(np.uint8) * 255) if th.use_pixel and sc.pix_prob is not None else None
        kept.append(StripResult(r, heat, gt, pred, th.pixel, coarse))
