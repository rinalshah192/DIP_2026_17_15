"""
Synthetic AFID-like benchmark generator (development and tests only).
ECE501 - Digital Image Processing (Group 15)

Creates 256x4096 woven-texture strips for 7 synthetic weave types, with injected
defects and pixel masks. Results on this data are NOT AFID results; the pipeline
only uses it when asked to explicitly (--synthetic).
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from fabric_defect.data.afid import STRIP_HEIGHT, STRIP_WIDTH

# Weave labels of the synthetic textures. Real AFID fabrics are identified by
# their filename code (fabric_00 .. fabric_06); AFID publishes no weave names.
FABRIC_TYPES = (
    "Fabric01_Plain",
    "Fabric02_Twill",
    "Fabric03_Satin",
    "Fabric04_Rib",
    "Fabric05_Basket",
    "Fabric06_Jacquard",
    "Fabric07_Herringbone",
)


def generate_synthetic_aitex_benchmark(
    base_dir: Path,
    samples_per_type: int = 3,
    height: int = STRIP_HEIGHT,
    width: int = STRIP_WIDTH
) -> pd.DataFrame:
    """
    Generates realistic 256x4096 woven fabric strips across all 7 fabric weave types,
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
                "mask_paths": [],
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

        for defect_idx, (defect_name, relative_pos) in enumerate(defects_to_inject):
            img_name = f"{fabric_name}_{defect_name}.png"
            mask_name = f"{fabric_name}_{defect_name}_mask.png"
            img_path = defective_dir / img_name
            mask_path = masks_dir / mask_name

            clean_strip = _synthesize_fabric_texture(
                fabric_type=fabric_name,
                height=height,
                width=width,
                seed=fabric_idx * 500 + defect_idx  # str hash() is salted per process
            )

            defective_strip, mask = _inject_defect(
                clean_strip,
                defect_type=defect_name,
                relative_pos=relative_pos,
                seed=fabric_idx * 500 + defect_idx
            )

            cv2.imwrite(str(img_path), defective_strip)
            cv2.imwrite(str(mask_path), mask)

            records.append({
                "image_path": str(img_path),
                "mask_paths": [str(mask_path)],
                "fabric_type": fabric_name,
                "is_defective": True,
                "defect_type": defect_name,
                "filename": img_name
            })

    df = pd.DataFrame(records)
    df.to_csv(base_dir / "manifest.csv", index=False)
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


def _inject_defect(
    clean_img: np.ndarray,
    defect_type: str,
    relative_pos: float = 0.5,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject a realistic textile defect into a clean strip and generate binary ground-truth mask."""
    rng = np.random.default_rng(seed)
    H, W = clean_img.shape
    defective = clean_img.copy()
    mask = np.zeros((H, W), dtype=np.uint8)

    # Defects are spread along the long (fabric-width) axis
    center_y = H // 2
    center_x = int(W * relative_pos)

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
            noise = rng.integers(40, 220, patch.shape, dtype=np.uint8)
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
