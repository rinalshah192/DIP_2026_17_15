"""
Repository paths. Not tunables, so they live outside the run config.
"""

from pathlib import Path

# src/fabric_defect/core/paths.py -> repository root is four levels up
ROOT_DIR = Path(__file__).resolve().parents[4]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "Results"
