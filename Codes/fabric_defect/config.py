"""
Pipeline configuration.
ECE501 - Digital Image Processing (Group 15)

Every tunable parameter lives in one frozen dataclass tree. Defaults reproduce the
mid-semester settings; a TOML file (see Codes/configs/default.toml) can override
any subset of them. The resolved config is saved next to every run's results.
"""

import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Repository paths (not tunables)
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RESULTS_DIR = ROOT_DIR / "Results"

# AFID line-scan strips are 256 rows (fabric travel) x 4096 columns (fabric width)
STRIP_HEIGHT = 256
STRIP_WIDTH = 4096

# Weave labels used only by the synthetic benchmark generator.
# Real AFID fabrics are identified by their filename code (fabric_00 .. fabric_06).
FABRIC_TYPES = (
    "Fabric01_Plain",
    "Fabric02_Twill",
    "Fabric03_Satin",
    "Fabric04_Rib",
    "Fabric05_Basket",
    "Fabric06_Jacquard",
    "Fabric07_Herringbone",
)


@dataclass(frozen=True)
class PreprocessConfig:
    median_ksize: int = 3  # 0 disables the median filter
    gaussian_ksize: int = 0  # 0 disables the Gaussian filter
    gaussian_sigma: float = 1.0
    contrast: str = "clahe"  # "clahe", "histeq" or "none"
    clahe_clip_limit: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)
    illumination_sigma: float = 50.0  # 0 disables high-pass illumination flattening
    selvage_margin: int = 40  # columns trimmed inside the detected fabric ROI

    def __post_init__(self):
        if self.median_ksize and self.median_ksize % 2 == 0:
            raise ValueError("median_ksize must be odd (or 0)")
        if self.gaussian_ksize and self.gaussian_ksize % 2 == 0:
            raise ValueError("gaussian_ksize must be odd (or 0)")
        if self.contrast not in ("clahe", "histeq", "none"):
            raise ValueError(f"unknown contrast mode {self.contrast!r}")


@dataclass(frozen=True)
class PatchConfig:
    size: int = 256
    stride: int = 128  # 50 % overlap
    defect_min_pixels: int = 25  # a patch is labelled defective above this many mask pixels

    def __post_init__(self):
        if not 0 < self.stride <= self.size:
            raise ValueError("stride must be in (0, size]")


@dataclass(frozen=True)
class GLCMConfig:
    distances: Tuple[int, ...] = (1, 2, 3)
    angles_deg: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    levels: int = 64
    properties: Tuple[str, ...] = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")


@dataclass(frozen=True)
class LBPConfig:
    scales: Tuple[Tuple[int, int], ...] = ((8, 1), (16, 2))  # (points P, radius R), uniform riu2


@dataclass(frozen=True)
class ModelConfig:
    scoring: str = "hybrid"  # "hybrid", "relative" or "mahalanobis"
    regularization: float = 1e-4  # lambda in Sigma + lambda * tr(Sigma)/D * I
    k: float = 2.5  # tau = mu + k * sigma (mahalanobis) or robust-z cut-off (hybrid/relative)
    calibration_fraction: float = 0.25  # share of defect-free images held out to calibrate tau

    def __post_init__(self):
        if self.scoring not in ("hybrid", "relative", "mahalanobis"):
            raise ValueError(f"unknown scoring method {self.scoring!r}")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in [0, 1)")


@dataclass(frozen=True)
class SegmentConfig:
    kernel_size: int = 5
    min_area: int = 50


@dataclass(frozen=True)
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    glcm: GLCMConfig = field(default_factory=GLCMConfig)
    lbp: LBPConfig = field(default_factory=LBPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    max_test_samples: int = 14
    seed: int = 0

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PipelineConfig":
        """Defaults, overridden by the sections/keys present in a TOML file."""
        cfg = cls()
        if path is None:
            return cfg
        data = tomllib.loads(Path(path).read_text())
        updates = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data.pop(f.name)
            current = getattr(cfg, f.name)
            if isinstance(value, dict):
                # replace() raises TypeError on unknown keys, which is what we want
                updates[f.name] = replace(current, **{k: _freeze(v) for k, v in value.items()})
            else:
                updates[f.name] = value
        if data:
            raise ValueError(f"Unknown config keys in {path}: {sorted(data)}")
        return replace(cfg, **updates)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _freeze(value: Any) -> Any:
    """TOML arrays come back as lists; the dataclasses are frozen and expect tuples."""
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value
