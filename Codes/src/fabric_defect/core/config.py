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
class GaborConfig:
    orientations_deg: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    wavelengths: Tuple[float, ...] = (4.0, 8.0, 16.0)  # px; AFID yarn period is roughly 4-8 px
    bandwidth: float = 1.0  # octaves; sets sigma from the wavelength
    gamma: float = 0.5  # spatial aspect ratio of the Gaussian envelope


FEATURE_BLOCKS = ("glcm", "lbp", "gabor")


@dataclass(frozen=True)
class FeatureConfig:
    use: Tuple[str, ...] = ("glcm", "lbp")  # blocks concatenated into the descriptor, in this order

    def __post_init__(self):
        if not self.use or any(b not in FEATURE_BLOCKS for b in self.use):
            raise ValueError(f"features.use must be a non-empty subset of {FEATURE_BLOCKS}")


@dataclass(frozen=True)
class ModelConfig:
    scoring: str = "hybrid"  # "hybrid", "relative", "mahalanobis", "strip_mahalanobis", or baselines "ocsvm", "iforest"
    regularization: float = 1e-4  # lambda in Sigma + lambda * tr(Sigma)/D * I
    k: float = 2.5  # tau = mu + k * sigma (mahalanobis) or robust-z cut-off (hybrid/relative)
    calibration_fraction: float = 0.25  # share of defect-free images held out to calibrate tau
    reference: str = "per_fabric"  # "per_fabric" (adaptive: one model and tau per fabric) or "global" (fixed)
    ocsvm_nu: float = 0.1  # One-Class SVM: upper bound on the training outlier fraction
    iforest_trees: int = 200

    def __post_init__(self):
        if self.scoring not in ("hybrid", "relative", "mahalanobis", "strip_mahalanobis", "ocsvm", "iforest"):
            raise ValueError(f"unknown scoring method {self.scoring!r}")
        if self.reference not in ("per_fabric", "global"):
            raise ValueError(f"unknown reference {self.reference!r}")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in [0, 1)")


@dataclass(frozen=True)
class RefineConfig:
    enabled: bool = False
    size: int = 64  # fine window side, px
    stride: int = 32
    k: float = 2.5  # tau_fine = mu + k * sigma of held-out fine-window Mahalanobis distances

    def __post_init__(self):
        if not 0 < self.stride <= self.size:
            raise ValueError("refine.stride must be in (0, refine.size]")


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
    gabor: GaborConfig = field(default_factory=GaborConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    max_test_samples: int = 0  # 0 = every defective strip; n = n strips stratified by fabric
    seed: int = 0

    def __post_init__(self):
        if self.refine.size >= self.patch.size:
            raise ValueError("refine.size must be smaller than patch.size")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PipelineConfig":
        """Defaults, overridden by the sections/keys present in a TOML file."""
        if path is None:
            return cls()
        return cls().with_overrides(tomllib.loads(Path(path).read_text()), source=str(path))

    def with_overrides(self, data: Dict[str, Any], source: str = "overrides") -> "PipelineConfig":
        """A copy with the given {section: {key: value}} / {key: value} entries replaced."""
        cfg, data, cls = self, dict(data), type(self)
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
            raise ValueError(f"Unknown config keys in {source}: {sorted(data)}")
        return replace(cfg, **updates)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _freeze(value: Any) -> Any:
    """TOML arrays come back as lists; the dataclasses are frozen and expect tuples."""
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value
