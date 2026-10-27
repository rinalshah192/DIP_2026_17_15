"""
Command line entry point.

    python -m fabric_defect run                      # AFID in data/raw, results in Results/
    python -m fabric_defect run --config my.toml --out Results/runs/exp1
    python -m fabric_defect run --synthetic --out /tmp/smoke
    python -m fabric_defect ablate --grid Codes/configs/experiments/ablation.toml --out Results/ablation
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from fabric_defect.core.config import PipelineConfig
from fabric_defect.core.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR

log = logging.getLogger("fabric_defect")

SCORING = ["hybrid", "relative", "mahalanobis", "strip_mahalanobis", "ocsvm", "iforest"]


def _common(p: argparse.ArgumentParser, default_out: Path) -> None:
    """Flags shared by every subcommand."""
    p.add_argument("--data-dir", type=Path, default=None,
                   help=f"AFID root with NODefect_images/, Defect_images/, Mask_images/ (default {RAW_DATA_DIR})")
    p.add_argument("--out", type=Path, default=default_out, help=f"output directory (default {default_out})")
    p.add_argument("--synthetic", action="store_true",
                   help="generate and use synthetic strips (smoke test, not AFID results)")
    p.add_argument("--max-test", type=int, default=None, help="override number of defective strips evaluated (0 = all)")
    p.add_argument("--workers", type=int, default=1, help="processes for feature extraction (default 1)")
    p.add_argument("--cache-dir", type=Path, default=PROCESSED_DATA_DIR,
                   help=f"feature cache folder (default {PROCESSED_DATA_DIR})")
    p.add_argument("--no-cache", action="store_true", help="always recompute features")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fabric_defect", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fit reference models and evaluate defective strips")
    _common(run, RESULTS_DIR)
    run.add_argument("--config", type=Path, default=None, help="TOML config overriding the defaults")
    run.add_argument("--seed", type=int, default=None, help="override config seed")
    run.add_argument("--scoring", choices=SCORING, default=None, help="override anomaly scoring method")

    ablate = sub.add_parser("ablate", help="run every variant of an ablation grid on shared features")
    _common(ablate, RESULTS_DIR / "ablation")
    ablate.add_argument("--grid", type=Path, required=True, help="grid TOML (see Codes/configs/experiments/)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    data_dir = args.data_dir or (args.out / "synthetic_data" if args.synthetic else RAW_DATA_DIR)
    # Synthetic strips are regenerated per run, so caching them only fills the cache
    cache_dir = None if args.no_cache or args.synthetic else args.cache_dir

    # Imported here so --help stays fast (cv2/skimage/matplotlib are slow to import)
    from fabric_defect.pipelines.ablation import load_grid, run_ablation
    from fabric_defect.pipelines.benchmark import run_full_pipeline

    # Config problems are user errors (exit 2); anything raised later is a bug and should surface
    try:
        cfg = load_grid(args.grid) and None if args.command == "ablate" else _run_config(args)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log.error("Invalid config: %s", exc)
        return 2
    try:
        if args.command == "ablate":
            results = run_ablation(args.grid, data_dir, args.out, synthetic=args.synthetic,
                                   cache_dir=cache_dir, workers=args.workers, max_test=args.max_test)
        else:
            results = run_full_pipeline(cfg, data_dir=data_dir, out_dir=args.out, synthetic=args.synthetic,
                                        cache_dir=cache_dir, workers=args.workers)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    return 0 if len(results) else 1


def _run_config(args) -> PipelineConfig:
    cfg = PipelineConfig.load(args.config)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.max_test is not None:
        cfg = replace(cfg, max_test_samples=args.max_test)
    if args.scoring is not None:
        cfg = replace(cfg, model=replace(cfg.model, scoring=args.scoring))
    return cfg


if __name__ == "__main__":
    sys.exit(main())
