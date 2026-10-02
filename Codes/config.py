"""
Global Configuration for Fabric Surface Defect Detection Pipeline.
ECE501 - Digital Image Processing (Group 15)
"""

from pathlib import Path
import numpy as np

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "Results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = ROOT_DIR / "reports"

# Geometry specifications for AITEX Fabric Image Database
STRIP_HEIGHT = 4096
STRIP_WIDTH = 256
PATCH_SIZE = 256
PATCH_STRIDE = 128  # 50% overlap for fine spatial localization (or 256 for non-overlapping)

# Preprocessing parameters
MEDIAN_KERNEL_SIZE = 3
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# GLCM Texture Parameters
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS = 64  # Intensity quantization to suppress noise and optimize matrix size
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

# Multi-scale LBP Parameters (P: points, R: radius)
LBP_CONFIGS = [
    {"points": 8, "radius": 1, "method": "uniform"},
    {"points": 16, "radius": 2, "method": "uniform"}
]

# Anomaly Scoring & Localization
MAHALANOBIS_REGULARIZATION = 1e-4
ADAPTIVE_THRESHOLD_K = 2.5  # tau = mu_score + k * sigma_score
MIN_DEFECT_AREA_PIXELS = 50  # Remove isolated false alarm components smaller than this

# Defect and Fabric Metadata (AITEX AFID benchmark)
FABRIC_TYPES = [
    "Fabric01_Plain",
    "Fabric02_Twill",
    "Fabric03_Satin",
    "Fabric04_Rib",
    "Fabric05_Basket",
    "Fabric06_Jacquard",
    "Fabric07_Herringbone"
]

DEFECT_TYPES = [
    "broken_end",
    "broken_pick",
    "broken_yarn",
    "weft_curling",
    "fuzzy_ball",
    "cut_selvage",
    "crease",
    "warp_ball",
    "knot",
    "contamination",
    "nep",
    "weft_crack"
]
