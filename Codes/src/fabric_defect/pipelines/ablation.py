"""
Ablation runner.
ECE501 - Digital Image Processing (Group 15)

Runs a grid of named config variants (Codes/configs/experiments/*.toml) on the
same catalog, the same image-level fit/calibration split and the same cached
window features, so that each row of the summary differs from the base only in
the overrides it names.
"""

import logging
import re
import tomllib
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from fabric_defect.core.config import PipelineConfig
from fabric_defect.evaluation.metrics import summarize
from fabric_defect.features.cache import load_features
from fabric_defect.pipelines.benchmark import (
    evaluate,
    feature_scales,
    fit_models,
    fit_refiners,
    load_manifest,
    select_test,
    split_by_fabric,
    write_run_info,
)

log = logging.getLogger(__name__)


def load_grid(path: Path) -> Tuple[PipelineConfig, List[Tuple[str, PipelineConfig]]]:
    """Base config and the (name, config) variants of a grid file.

    Grid format: optional `base = "<toml relative to the grid>"`, then
    [[variant]] tables with `name` and an `overrides` table in config shape.
    """
    path = Path(path)
    data = tomllib.loads(path.read_text())
    base = PipelineConfig.load(path.parent / data["base"]) if "base" in data else PipelineConfig()
    variants, seen = [], set()
    for v in data.get("variant", []):
        name = v["name"]
        if name in seen:
            raise ValueError(f"duplicate variant name {name!r} in {path}")
        seen.add(name)
        variants.append((name, base.with_overrides(v.get("overrides", {}), source=f"{path}:{name}")))
    if not variants:
        raise ValueError(f"{path} defines no [[variant]]")
    return base, variants


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def run_ablation(
    grid: Path,
    data_dir: Path,
    out_dir: Path,
    synthetic: bool = False,
    cache_dir: Optional[Path] = None,
    workers: int = 1,
    max_test: Optional[int] = None
) -> pd.DataFrame:
    base, variants = load_grid(grid)
    if max_test is not None:
        variants = [(n, c.with_overrides({"max_test_samples": max_test})) for n, c in variants]
    manifest = load_manifest(data_dir, synthetic)
    clean = manifest[~manifest["is_defective"]]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_info(out_dir, base, Path(data_dir), synthetic, grid=str(grid),
                   variants={n: c.to_dict() for n, c in variants})

    rows = []
    for i, (name, cfg) in enumerate(variants, 1):
        test = select_test(manifest[manifest["is_defective"]], cfg.max_test_samples)
        split = split_by_fabric(clean, cfg)
        feats = load_features(clean["image_path"].tolist() + test["image_path"].tolist(),
                              cfg, feature_scales(cfg), cache_dir, workers)
        results, _ = evaluate(test, feats, fit_models(feats, split, cfg), cfg,
                              refiners=fit_refiners(feats, split, cfg))
        results.to_csv(out_dir / f"{slug(name)}.csv", index=False)
        row = {"variant": name, "features": "+".join(cfg.features.use), "scoring": cfg.model.scoring,
               "reference": cfg.model.reference, "refine": cfg.refine.enabled, **summarize(results)}
        rows.append(row)
        log.info("[%d/%d] %s: patch F1 %.2f%%, pixel IoU %.2f%% (n=%d)", i, len(variants), name,
                 100 * row["patch_f1"], 100 * row["pixel_iou"], row["n_images"])

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    log.info("Saved %s", out_dir / "summary.csv")
    return summary
