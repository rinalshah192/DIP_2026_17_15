"""
Dataset Management & AITEX Fabric Image Database (AFID) Handler.
ECE501 - Digital Image Processing (Group 15)

Provides:
1. Dataset cataloguing and structure inspection for AFID.
2. Image & mask pair loader.
3. High-fidelity synthetic AFID generator creating realistic 4096x256 woven fabric strips
   with ground-truth defect masks for complete reproducibility.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import pandas as pd

from Codes.config import (
    RAW_DATA_DIR,
    STRIP_HEIGHT,
    STRIP_WIDTH,
    FABRIC_TYPES,
    DEFECT_TYPES
)


class AITEXDataset:
    """Manages the AITEX Fabric Image Database directory and metadata."""

    def __init__(self, base_dir: Path = RAW_DATA_DIR):
        self.base_dir = Path(base_dir)
        self.defect_free_dir = self.base_dir / "defect_free"
        self.defective_dir = self.base_dir / "defective"
        self.masks_dir = self.base_dir / "masks"

        # Ensure directory structure exists
        self.defect_free_dir.mkdir(parents=True, exist_ok=True)
        self.defective_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    def scan_catalog(self) -> pd.DataFrame:
        """Scan directories and build inventory of available images and masks."""
        records = []

        # Check for original AITEX directory layout
        orig_nodefect = self.base_dir / "NODefect_images"
        orig_defect = self.base_dir / "Defect_images"
        orig_masks = self.base_dir / "Mask_images"

        if orig_nodefect.exists() and orig_defect.exists():
            # Scan original AITEX clean images
            for f in orig_nodefect.rglob("*.png"):
                fabric_type = self._infer_fabric_type(f.name)
                records.append({
                    "image_path": str(f),
                    "mask_path": None,
                    "fabric_type": fabric_type,
                    "is_defective": False,
                    "defect_type": "none",
                    "filename": f.name
                })

            # Scan original AITEX defective images
            for f in orig_defect.glob("*.png"):
                fabric_type = self._infer_fabric_type(f.name)
                defect_type = self._infer_defect_type(f.name)
                mask_path = orig_masks / f"{f.stem}_mask.png"
                if not mask_path.exists():
                    mask_path = orig_masks / f.name

                records.append({
                    "image_path": str(f),
                    "mask_path": str(mask_path) if mask_path.exists() else None,
                    "fabric_type": fabric_type,
                    "is_defective": True,
                    "defect_type": defect_type,
                    "filename": f.name
                })
        else:
            # Scan standardized layout
            for f in self.defect_free_dir.glob("*.png"):
                fabric_type = self._infer_fabric_type(f.name)
                records.append({
                    "image_path": str(f),
                    "mask_path": None,
                    "fabric_type": fabric_type,
                    "is_defective": False,
                    "defect_type": "none",
                    "filename": f.name
                })

            for f in self.defective_dir.glob("*.png"):
                fabric_type = self._infer_fabric_type(f.name)
                defect_type = self._infer_defect_type(f.name)
                mask_path = self.masks_dir / f"{f.stem}_mask.png"
                if not mask_path.exists():
                    mask_path = self.masks_dir / f.name

                records.append({
                    "image_path": str(f),
                    "mask_path": str(mask_path) if mask_path.exists() else None,
                    "fabric_type": fabric_type,
                    "is_defective": True,
                    "defect_type": defect_type,
                    "filename": f.name
                })

        df = pd.DataFrame(records)
        return df

    def _infer_fabric_type(self, filename: str) -> str:
        stem = Path(filename).stem
        parts = stem.split("_")
        # In original AITEX (e.g. 0001_000_02.png or 0001_002_00.png), fabric index is the last number
        if len(parts) >= 3 and parts[-1].isdigit():
            idx = int(parts[-1]) % len(FABRIC_TYPES)
            return FABRIC_TYPES[idx]

        for ft in FABRIC_TYPES:
            code = ft.split("_")[0]
            if code.lower() in filename.lower() or ft.lower() in filename.lower():
                return ft
        return "Fabric01_Plain"

    def _infer_defect_type(self, filename: str) -> str:
        stem = Path(filename).stem
        parts = stem.split("_")
        # In original AITEX (e.g. 0001_002_00.png), defect code is middle part
        if len(parts) >= 3 and parts[1].isdigit():
            defect_code_map = {
                "001": "broken_end",
                "002": "broken_pick",
                "003": "broken_yarn",
                "004": "weft_curling",
                "005": "fuzzy_ball",
                "006": "cut_selvage",
                "007": "crease",
                "008": "warp_ball",
                "009": "knot",
                "010": "contamination",
                "011": "nep",
                "012": "weft_crack"
            }
            code = parts[1].zfill(3)
            if code in defect_code_map:
                return defect_code_map[code]

        for dt in DEFECT_TYPES:
            if dt in filename.lower():
                return dt
        return "defect_general"


def generate_synthetic_aitex_benchmark(
    base_dir: Path = RAW_DATA_DIR,
    samples_per_type: int = 3,
    height: int = STRIP_HEIGHT,
    width: int = STRIP_WIDTH
) -> pd.DataFrame:
    """
    Generates realistic 4096x256 woven fabric strips across all 7 fabric weave types,
    including illumination gradients, thread micro-structure, and realistic textile defects
    paired with pixel-level ground truth masks.

    Ensures that the entire classical DIP pipeline can be verified reproducibly.
    """
    base_dir = Path(base_dir)
    defect_free_dir = base_dir / "defect_free"
    defective_dir = base_dir / "defective"
    masks_dir = base_dir / "masks"

    defect_free_dir.mkdir(parents=True, exist_ok=True)
    defective_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for fabric_idx, fabric_name in enumerate(FABRIC_TYPES):
        # 1. Generate Defect-Free reference strips
        for i in range(samples_per_type):
            img_name = f"{fabric_name}_clean_{i+1:02d}.png"
            img_path = defect_free_dir / img_name

            strip = _synthesize_fabric_texture(
                fabric_type=fabric_name,
                height=height,
                width=width,
                seed=fabric_idx * 100 + i
            )
            cv2.imwrite(str(img_path), strip)
            records.append({
                "image_path": str(img_path),
                "mask_path": None,
                "fabric_type": fabric_name,
                "is_defective": False,
                "defect_type": "none",
                "filename": img_name
            })

        # 2. Generate Defective strips with paired pixel ground truth masks
        defects_to_inject = [
            ("broken_pick", 0.5),   # Horizontal missing thread
            ("broken_end", 0.3),    # Vertical missing thread
            ("fuzzy_ball", 0.7),    # Circular textured pilling
            ("contamination", 0.2), # Dark oil spot
            ("cut_selvage", 0.85)   # Edge tear
        ]

        for defect_name, relative_pos in defects_to_inject:
            img_name = f"{fabric_name}_{defect_name}.png"
            mask_name = f"{fabric_name}_{defect_name}_mask.png"
            img_path = defective_dir / img_name
            mask_path = masks_dir / mask_name

            clean_strip = _synthesize_fabric_texture(
                fabric_type=fabric_name,
                height=height,
                width=width,
                seed=fabric_idx * 500 + hash(defect_name) % 100
            )

            defective_strip, mask = _inject_defect(
                clean_strip,
                defect_type=defect_name,
                relative_pos=relative_pos
            )

            cv2.imwrite(str(img_path), defective_strip)
            cv2.imwrite(str(mask_path), mask)

            records.append({
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "fabric_type": fabric_name,
                "is_defective": True,
                "defect_type": defect_name,
                "filename": img_name
            })

    df = pd.DataFrame(records)
    summary_path = base_dir / "aitex_manifest.csv"
    df.to_csv(summary_path, index=False)
    print(f"Catalogued {len(df)} images in {summary_path}")
    return df


def _synthesize_fabric_texture(fabric_type: str, height: int, width: int, seed: int = 42) -> np.ndarray:
    """Mathematical 2D synthesis of periodic woven structures with illumination gradient."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, height, height, endpoint=False)
    x = np.linspace(0, width, width, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Base yarn spatial frequency (approx. 4 to 8 pixels per thread)
    freq_warp = 2 * np.pi / 6.0
    freq_weft = 2 * np.pi / 6.0

    if "Plain" in fabric_type:
        # Cross weave: cos(x) * cos(y)
        tex = np.cos(freq_weft * X) * np.cos(freq_warp * Y)
    elif "Twill" in fabric_type:
        # Diagonal weave: cos(x + y)
        tex = 0.7 * np.cos(freq_weft * X + freq_warp * Y) + 0.3 * np.cos(freq_weft * X - 0.5 * freq_warp * Y)
    elif "Satin" in fabric_type:
        # Subtle texture with high smoothness
        tex = 0.4 * np.cos(freq_weft * X * 0.5) * np.sin(freq_warp * Y * 0.5)
    elif "Rib" in fabric_type:
        # Strong horizontal cord structure
        tex = 0.8 * np.cos(freq_warp * Y * 0.6) + 0.2 * np.cos(freq_weft * X)
    elif "Basket" in fabric_type:
        # 2x2 yarn block grid
        tex = np.sign(np.cos(freq_weft * X * 0.5)) * np.sign(np.cos(freq_warp * Y * 0.5))
    elif "Jacquard" in fabric_type:
        # Multi-frequency modulated weave
        tex = 0.5 * np.cos(freq_weft * X) * np.cos(freq_warp * Y) + 0.5 * np.sin(freq_weft * X * 0.2 + freq_warp * Y * 0.3)
    else:  # Herringbone / Structured
        # Chevron zig-zag pattern
        zigzag_x = X + 8.0 * np.sign(np.sin(2 * np.pi * Y / 120.0))
        tex = np.cos(freq_weft * zigzag_x + freq_warp * Y)

    # Add yarn micro-roughness (Gaussian noise)
    noise = rng.normal(0, 0.15, (height, width))
    texture_field = tex + noise

    # Add realistic longitudinal illumination gradient (shadow fall-off along strip)
    lighting_gradient = 1.0 - 0.15 * (Y / height) - 0.08 * ((X - width/2) / width)**2

    # Map to [0, 255] uint8
    norm_field = (texture_field - texture_field.min()) / (texture_field.max() - texture_field.min())
    final_img = (norm_field * 200 + 35) * lighting_gradient
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    return final_img


def _inject_defect(clean_img: np.ndarray, defect_type: str, relative_pos: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Inject a realistic textile defect into a clean strip and generate binary ground-truth mask."""
    H, W = clean_img.shape
    defective = clean_img.copy()
    mask = np.zeros((H, W), dtype=np.uint8)

    center_y = int(H * relative_pos)
    center_x = int(W * 0.5)

    if defect_type == "broken_pick":
        # Missing horizontal weft thread across width
        y_start = center_y - 3
        y_end = center_y + 3
        defective[y_start:y_end, :] = (defective[y_start:y_end, :].astype(np.float32) * 0.3).astype(np.uint8)
        mask[y_start:y_end, :] = 255

    elif defect_type == "broken_end":
        # Missing vertical warp thread along a segment
        x_start = center_x - 3
        x_end = center_x + 3
        y_start = max(0, center_y - 200)
        y_end = min(H, center_y + 200)
        defective[y_start:y_end, x_start:x_end] = (defective[y_start:y_end, x_start:x_end].astype(np.float32) * 0.35).astype(np.uint8)
        mask[y_start:y_end, x_start:x_end] = 255

    elif defect_type == "fuzzy_ball":
        # Fiber clump / pilling
        radius = 24
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        # Disrupt texture with high variance clutter
        patch = defective[center_y - radius : center_y + radius, center_x - radius : center_x + radius]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            noise = np.random.randint(40, 220, patch.shape, dtype=np.uint8)
            defective[center_y - radius : center_y + radius, center_x - radius : center_x + radius] = noise

    elif defect_type == "contamination":
        # Oil / grease stain (irregular darkened region)
        pts = np.array([
            [center_x - 30, center_y - 20],
            [center_x + 40, center_y - 35],
            [center_x + 60, center_y + 25],
            [center_x - 10, center_y + 40]
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        # Darken region smoothly
        defective[mask == 255] = (defective[mask == 255].astype(np.float32) * 0.45).astype(np.uint8)

    elif defect_type == "cut_selvage":
        # Laceration / tear along edge
        pts = np.array([
            [0, center_y - 60],
            [45, center_y],
            [0, center_y + 60]
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        defective[mask == 255] = 20  # Open void

    else:
        # Generic defect
        cv2.rectangle(mask, (center_x - 20, center_y - 20), (center_x + 20, center_y + 20), 255, -1)
        defective[mask == 255] = 230

    return defective, mask
