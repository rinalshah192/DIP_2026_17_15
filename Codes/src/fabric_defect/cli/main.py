"""
Command line entry point.

    python -m fabric_defect run                      # AFID in data/raw, results in Results/
    python -m fabric_defect run --config my.toml --out Results/runs/exp1
    python -m fabric_defect run --synthetic --out /tmp/smoke
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fabric_defect", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fit reference models and evaluate defective strips")
    run.add_argument("--data-dir", type=Path, default=None,
                     help=f"AFID root with NODefect_images/, Defect_images/, Mask_images/ (default {RAW_DATA_DIR})")
    run.add_argument("--out", type=Path, default=RESULTS_DIR, help="output directory (default Results/)")
    run.add_argument("--config", type=Path, default=None, help="TOML config overriding the defaults")
    run.add_argument("--synthetic", action="store_true",
                     help="generate and use synthetic strips (smoke test, not AFID results)")
    run.add_argument("--seed", type=int, default=None, help="override config seed")
    run.add_argument("--max-test", type=int, default=None, help="override number of defective strips evaluated (0 = all)")
    run.add_argument("--scoring", choices=["hybrid", "relative", "mahalanobis"], default=None,
                     help="override anomaly scoring method")
    run.add_argument("--workers", type=int, default=1, help="processes for feature extraction (default 1)")
    run.add_argument("--cache-dir", type=Path, default=PROCESSED_DATA_DIR,
                     help=f"feature cache folder (default {PROCESSED_DATA_DIR})")
    run.add_argument("--no-cache", action="store_true", help="always recompute features")
    run.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    # Imported here so --help stays fast (cv2/skimage/matplotlib are slow to import)
    from fabric_defect.pipelines.benchmark import run_full_pipeline

    try:
        cfg = PipelineConfig.load(args.config)
    except (OSError, ValueError, TypeError) as exc:
        log.error("Invalid config %s: %s", args.config, exc)
        return 2
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.max_test is not None:
        cfg = replace(cfg, max_test_samples=args.max_test)
    if args.scoring is not None:
        cfg = replace(cfg, model=replace(cfg.model, scoring=args.scoring))

    data_dir = args.data_dir or (args.out / "synthetic_data" if args.synthetic else RAW_DATA_DIR)
    try:
        # Synthetic strips are regenerated per run, so caching them only fills the cache
        cache_dir = None if args.no_cache or args.synthetic else args.cache_dir
        results = run_full_pipeline(cfg, data_dir=data_dir, out_dir=args.out, synthetic=args.synthetic,
                                    cache_dir=cache_dir, workers=args.workers)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    return 0 if len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
