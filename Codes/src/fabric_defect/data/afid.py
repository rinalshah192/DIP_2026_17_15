"""
Dataset Management & AITEX Fabric Image Database (AFID) Handler.
ECE501 - Digital Image Processing (Group 15)

Provides:
1. Dataset cataloguing for the published AFID layout
   (NODefect_images/, Defect_images/, Mask_images/).
2. Filename parsing with the official AFID defect codes.
3. Multi-part ground-truth mask loading.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

from fabric_defect.core.paths import RAW_DATA_DIR

log = logging.getLogger(__name__)

# AFID line-scan strips are 256 rows (fabric travel) x 4096 columns (fabric width)
STRIP_HEIGHT = 256
STRIP_WIDTH = 4096

# Official AFID defect codes (Silvestre-Blanes et al., AUTEX Res. J. 2019).
# 000 marks a defect-free image.
DEFECT_CODES = {
    "000": "none",
    "002": "broken_end",
    "006": "broken_yarn",
    "010": "broken_pick",
    "016": "weft_curling",
    "019": "fuzzy_ball",
    "022": "cut_selvage",
    "023": "crease",
    "025": "warp_ball",
    "027": "knot",
    "029": "contamination",
    "030": "nep",
    "036": "weft_crack",
}

# AFID filenames are nnnn_ddd_ff: image id, defect code, fabric code (00-06)
_AFID_NAME = re.compile(r"^(\d{4})_(\d{3})_(\d{2})$")


def parse_afid_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Return (fabric_id, defect_name) for an AFID filename, or None if it is not one."""
    m = _AFID_NAME.match(Path(filename).stem)
    if m is None:
        return None
    _, code, fabric = m.groups()
    return f"fabric_{fabric}", DEFECT_CODES.get(code, f"unknown_{code}")


class AITEXDataset:
    """Catalogues the AITEX Fabric Image Database in its published directory layout."""

    def __init__(self, base_dir: Path = RAW_DATA_DIR):
        self.base_dir = Path(base_dir)
        self.normal_dir = self.base_dir / "NODefect_images"
        self.defect_dir = self.base_dir / "Defect_images"
        self.mask_dir = self.base_dir / "Mask_images"

    def scan_catalog(self) -> pd.DataFrame:
        """Build a manifest (one row per image) of defect-free and defective strips."""
        missing = [d.name for d in (self.normal_dir, self.defect_dir, self.mask_dir) if not d.is_dir()]
        if missing:
            raise FileNotFoundError(
                f"AFID not found under {self.base_dir}: missing {', '.join(missing)}. "
                "Download AFID there, or run with --synthetic for a smoke test."
            )

        records = []
        # Defect-free images sit in per-fabric sub-folders
        for f in sorted(self.normal_dir.rglob("*.png")):
            if parse_afid_filename(f.name) is None:
                # AFID ships one such file: NODefect_images/2608691-202020u/0018_00_01.png
                log.warning("Skipping %s: not an nnnn_ddd_ff AFID filename", f.name)
                continue
            records.append({
                "image_path": str(f),
                "mask_paths": [],
                "fabric_type": self._infer_fabric_type(f.name),
                "is_defective": False,
                "defect_type": "none",
                "filename": f.name
            })
        for f in sorted(self.defect_dir.glob("*.png")):
            if parse_afid_filename(f.name) is None:
                log.warning("Skipping %s: not an nnnn_ddd_ff AFID filename", f.name)
                continue
            records.append({
                "image_path": str(f),
                "mask_paths": find_masks(self.mask_dir, f.stem),
                "fabric_type": self._infer_fabric_type(f.name),
                "is_defective": True,
                "defect_type": self._infer_defect_type(f.name),
                "filename": f.name
            })
        return pd.DataFrame(records)

    def _infer_fabric_type(self, filename: str) -> str:
        parsed = parse_afid_filename(filename)
        return parsed[0] if parsed else "unknown"

    def _infer_defect_type(self, filename: str) -> str:
        parsed = parse_afid_filename(filename)
        return parsed[1] if parsed else "unknown"


def find_masks(mask_dir: Path, stem: str) -> List[str]:
    """All mask files for one image; some AFID defects ship as several parts (_mask, _mask1, ...)."""
    return sorted(str(p) for p in Path(mask_dir).glob(f"{stem}_mask*.png"))


def load_mask(paths: List[str], shape: Tuple[int, int]) -> np.ndarray:
    """Load and OR together every mask part into one binary mask (0 / 255) of the given shape."""
    if not paths:
        raise ValueError("No mask files given")
    out = np.zeros(shape, dtype=np.uint8)
    for path in paths:
        part = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if part is None:
            raise FileNotFoundError(f"Failed to read mask at {path}")
        if part.shape != tuple(shape):
            part = cv2.resize(part, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out[part > 127] = 255
    return out
