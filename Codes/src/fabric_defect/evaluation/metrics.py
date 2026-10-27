"""
Evaluation Metrics Harness for Fabric Defect Inspection.
ECE501 - Digital Image Processing (Group 15)

Undefined ratios (zero denominator) are reported as NaN, not 0, so that
"nothing to measure" is never averaged in as "complete miss". Summaries use
NaN-skipping means and report how many images each mean is taken over.

Computes standardized quantitative metrics:
1. Patch-level Classification: Accuracy, Precision, Recall, F1-Score, Specificity
2. Pixel-level Localization: Intersection over Union (IoU), Dice (F1), Pixel Precision, Recall
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def evaluate_patch_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute patch-level confusion matrix and classification metrics.

    Args:
        y_true: 1D array of ground truth labels (0=clean, 1=defective)
        y_pred: 1D array of predicted labels (0=clean, 1=defective)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = _ratio(tp + tn, tp + fp + tn + fn)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    # Count form of F1: defined (and 0) whenever there is any positive or prediction
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    specificity = _ratio(tn, tn + fp)

    return {
        "patch_accuracy": float(accuracy),
        "patch_precision": float(precision),
        "patch_recall": float(recall),
        "patch_f1": float(f1),
        "patch_specificity": float(specificity),
        "patch_tp": tp,
        "patch_fp": fp,
        "patch_tn": tn,
        "patch_fn": fn
    }


def evaluate_pixel_localization(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> Dict[str, float]:
    """
    Compute pixel-level segmentation accuracy and localization overlap.

    Args:
        pred_mask: 2D binary predicted mask (0 or 255)
        gt_mask: 2D binary ground truth mask (0 or 255)
    """
    p_bin = (pred_mask > 0).astype(bool)
    g_bin = (gt_mask > 0).astype(bool)

    intersection = np.logical_and(p_bin, g_bin).sum()
    union = np.logical_or(p_bin, g_bin).sum()
    p_count = p_bin.sum()
    g_count = g_bin.sum()

    # Empty ground truth -> recall undefined; empty prediction -> precision undefined;
    # both empty -> IoU/Dice undefined as well
    iou = _ratio(intersection, union)
    dice = _ratio(2 * intersection, p_count + g_count)
    precision = _ratio(intersection, p_count)
    recall = _ratio(intersection, g_count)

    return {
        "pixel_iou": iou,
        "pixel_dice": dice,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "gt_defect_pixels": int(g_count),
        "pred_defect_pixels": int(p_count),
        "pixel_tp": int(intersection),  # kept so pooled metrics can be summed over images
    }


SUMMARY_METRICS = (
    "patch_accuracy", "patch_precision", "patch_recall", "patch_f1", "patch_specificity",
    "pixel_iou", "pixel_dice", "pixel_precision", "pixel_recall",
)


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    """NaN-skipping means of the per-image metrics, pooled (micro) metrics, and the image count.

    A per-image mean gives a 50-pixel nep the same weight as a full-width crease
    and skips images where a ratio is undefined; pooled metrics sum TP/FP/FN
    over all images first, so they weight every pixel (or patch) equally.
    """
    out = {"n_images": int(len(df))}
    out.update({m: float(df[m].mean()) if len(df) else float("nan") for m in SUMMARY_METRICS})
    if len(df):
        tp, fp, fn = (float(df[f"patch_{k}"].sum()) for k in ("tp", "fp", "fn"))
        out.update(pooled_patch_precision=_ratio(tp, tp + fp), pooled_patch_recall=_ratio(tp, tp + fn),
                   pooled_patch_f1=_ratio(2 * tp, 2 * tp + fp + fn))
        ptp = float(df["pixel_tp"].sum())
        pred, gt = float(df["pred_defect_pixels"].sum()), float(df["gt_defect_pixels"].sum())
        out.update(pooled_pixel_iou=_ratio(ptp, pred + gt - ptp), pooled_pixel_dice=_ratio(2 * ptp, pred + gt),
                   pooled_pixel_precision=_ratio(ptp, pred), pooled_pixel_recall=_ratio(ptp, gt))
    return out


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


class EvaluationSuite:
    """Aggregates and formats quantitative evaluation results across multiple tests."""

    def __init__(self):
        self.records: List[Dict] = []

    def log_result(
        self,
        sample_id: str,
        fabric_type: str,
        defect_type: str,
        patch_metrics: Dict[str, float],
        pixel_metrics: Dict[str, float],
        descriptor_used: str = "GLCM+LBP"
    ):
        combined = {
            "sample_id": sample_id,
            "fabric_type": fabric_type,
            "defect_type": defect_type,
            "descriptor": descriptor_used,
            **patch_metrics,
            **pixel_metrics
        }
        self.records.append(combined)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def summary_table(self, by: str = "fabric_type") -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df

        metric_cols = [
            "patch_accuracy", "patch_precision", "patch_recall", "patch_f1",
            "pixel_iou", "pixel_dice", "pixel_precision", "pixel_recall"
        ]
        groups = df.groupby([by, "descriptor"])
        summary = groups[metric_cols].mean().reset_index()  # pandas mean skips NaN
        summary.insert(2, "n_images", groups.size().values)
        return summary
