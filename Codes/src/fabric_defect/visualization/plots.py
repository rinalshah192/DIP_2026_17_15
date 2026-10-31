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


def save_score_distributions(df: pd.DataFrame, out_file: Path) -> Path:
    """Image scores (log scale) of defect-free vs defective test strips per fabric, plus the ROC curve."""
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    fabrics = sorted(df["fabric_type"].unique())
    for i, fab in enumerate(fabrics):
        g = df[df["fabric_type"] == fab]
        for defective, colour, dx in ((False, "tab:blue", -0.15), (True, "tab:red", 0.15)):
            v = g.loc[g["is_defective"].astype(bool) == defective, "image_score"]
            ax[0].scatter(np.full(len(v), i + dx), v, s=14, alpha=0.7, color=colour,
                          label=("defective" if defective else "defect-free") if i == 0 else None)
    ax[0].axhline(1.0, color="black", lw=0.8, ls="--", label="max window = tau")
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(fabrics)), fabrics, rotation=30)
    ax[0].set_ylabel("image score  S = max window score / tau")
    ax[0].set_title("Image scores of test strips", fontweight="bold")
    ax[0].legend(loc="upper left", fontsize=8)

    y = df["is_defective"].astype(bool)
    if 0 < y.sum() < len(y):
        fpr, tpr, _ = roc_curve(y, df["image_score"])
        ax[1].plot(fpr, tpr, color="tab:red")
    ax[1].plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    ax[1].set_xlabel("false-alarm rate")
    ax[1].set_ylabel("detection rate")
    ax[1].set_title("Image-level ROC", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_file
