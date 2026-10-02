"""
Evaluation Metrics Harness for Fabric Defect Inspection.
ECE501 - Digital Image Processing (Group 15)

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

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

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

    # Edge case: both ground truth and prediction are defect-free (true negative strip)
    if union == 0:
        iou = 1.0
        dice = 1.0
        precision = 1.0
        recall = 1.0
    else:
        iou = float(intersection / union) if union > 0 else 0.0
        dice = float(2.0 * intersection / (p_count + g_count)) if (p_count + g_count) > 0 else 0.0
        precision = float(intersection / p_count) if p_count > 0 else 0.0
        recall = float(intersection / g_count) if g_count > 0 else 0.0

    return {
        "pixel_iou": iou,
        "pixel_dice": dice,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "gt_defect_pixels": int(g_count),
        "pred_defect_pixels": int(p_count)
    }


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

    def summary_table(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df

        metric_cols = [
            "patch_accuracy", "patch_precision", "patch_recall", "patch_f1",
            "pixel_iou", "pixel_dice", "pixel_precision", "pixel_recall"
        ]
        summary = df.groupby(["fabric_type", "descriptor"])[metric_cols].mean().reset_index()
        return summary
