"""
Figures for inspection and benchmark summaries.
ECE501 - Digital Image Processing (Group 15)
"""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: CI and servers have no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_inspection_figure(
    out_file: Path,
    sample_id: str,
    raw_img: np.ndarray,
    norm_img: np.ndarray,
    heatmap: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    threshold: float,
    defect_type: str,
    fabric_type: str,
    coarse_mask: Optional[np.ndarray] = None
) -> Path:
    """Stacked panels along the strip: raw, pre-processed, deviation, ground truth, (coarse,) prediction."""
    panels = [
        (raw_img, "gray", f"1. Raw strip ({raw_img.shape[0]} x {raw_img.shape[1]})"),
        (norm_img, "gray", "2. Pre-processed (illumination flattening + median + CLAHE)"),
        (heatmap, "inferno", "3. Texture deviation score"),
        (gt_mask, "Blues_r", f"4. Ground-truth mask ({defect_type})"),
    ]
    if coarse_mask is not None:
        panels.append((coarse_mask, "Oranges", f"5. Coarse 256x256 mask (threshold {threshold:.2f})"))
        panels.append((pred_mask, "Reds", "6. Refined 64x64 mask"))
    else:
        panels.append((pred_mask, "Reds", f"5. Predicted mask (threshold {threshold:.2f} + morphology)"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 1.8 * len(panels)), sharex=True)
    for ax, (img, cmap, title) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.axis("off")
        if cmap == "inferno":
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    fig.suptitle(f"{fabric_type} | {defect_type} | {sample_id}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_file


def save_summary_charts(df: pd.DataFrame, chart_file: Path) -> Path:
    """Bar charts of mean metrics per defect type and per fabric."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    def_summary = df.groupby("defect_type")[["patch_f1", "patch_accuracy", "pixel_iou"]].mean()
    def_summary.plot(kind="bar", ax=ax[0], colormap="viridis", edgecolor="black", width=0.7)
    ax[0].set_title("Mean metrics per defect type", fontweight="bold")

    fab_summary = df.groupby("fabric_type")[["pixel_iou", "pixel_dice"]].mean()
    fab_summary.plot(kind="bar", ax=ax[1], colormap="magma", edgecolor="black", width=0.6)
    ax[1].set_title("Pixel localization per fabric", fontweight="bold")

    for a in ax:
        a.set_ylabel("Score [0, 1]")
        a.set_ylim(0, 1.1)
        a.grid(axis="y", linestyle="--", alpha=0.5)
        a.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(chart_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return chart_file
