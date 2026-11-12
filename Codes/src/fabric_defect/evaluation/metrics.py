"""
Evaluation Metrics Harness for Fabric Defect Inspection.
ECE501 - Digital Image Processing (Group 15)

Undefined ratios (zero denominator) are reported as NaN, not 0, so that
"nothing to measure" is never averaged in as "complete miss". Summaries use
NaN-skipping means and report how many images each mean is taken over.

Computes standardized quantitative metrics:
1. Patch-level Classification: Accuracy, Precision, Recall, F1-Score, Specificity
2. Pixel-level Localization: Intersection over Union (IoU), Dice (F1), Pixel Precision, Recall
3. Image-level Detection: accuracy, precision, recall, specificity, F1 and AUROC over defective
   and defect-free test strips

Patch and pixel metrics are summarised over defective strips only; image-level
metrics need both kinds.
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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

    out = {
        "pixel_iou": iou,
        "pixel_dice": dice,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "gt_defect_pixels": int(g_count),
        "pred_defect_pixels": int(p_count),
        "pixel_tp": int(intersection),  # kept so pooled metrics can be summed over images
    }
    for r in TOLERANCES:
        out.update(tolerant_pixel_metrics(p_bin, g_bin, r))
    return out


TOLERANCES = (2, 3)  # px; thin-structure detection usually allows a small misalignment


def tolerant_pixel_metrics(pred: np.ndarray, gt: np.ndarray, r: int) -> Dict[str, float]:
    """Precision / recall / F1 with an r-pixel tolerance, as in thin-structure (crack) detection:
    a predicted pixel is correct if a ground-truth pixel lies within r px of it, and a ground-truth
    pixel is found if a prediction lies within r px of it."""
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near_gt = cv2.dilate(gt.astype(np.uint8), k) > 0
    near_pred = cv2.dilate(pred.astype(np.uint8), k) > 0
    tp_pred, tp_gt = int((pred & near_gt).sum()), int((gt & near_pred).sum())
    prec, rec = _ratio(tp_pred, int(pred.sum())), _ratio(tp_gt, int(gt.sum()))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else (float("nan") if np.isnan(prec + rec) else 0.0)
    return {f"pixel_tol{r}_precision": prec, f"pixel_tol{r}_recall": rec, f"pixel_tol{r}_f1": f1,
            f"pixel_tol{r}_tp_pred": tp_pred, f"pixel_tol{r}_tp_gt": tp_gt}


SUMMARY_METRICS = (
    "patch_accuracy", "patch_precision", "patch_recall", "patch_f1", "patch_specificity",
    "pixel_iou", "pixel_dice", "pixel_precision", "pixel_recall",
    "patch_auroc",
)


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    """NaN-skipping means of the per-image metrics, pooled (micro) metrics, and the image count.

    A per-image mean gives a 50-pixel nep the same weight as a full-width crease
    and skips images where a ratio is undefined; pooled metrics sum TP/FP/FN
    over all images first, so they weight every pixel (or patch) equally.
    """
    full = df
    if "is_defective" in df.columns:
        df = df[df["is_defective"].astype(bool)]
    out = {"n_images": int(len(df))}
    out.update({m: float(df[m].mean()) if len(df) and m in df else float("nan") for m in SUMMARY_METRICS})
    if len(df):
        tp, fp, fn = (float(df[f"patch_{k}"].sum()) for k in ("tp", "fp", "fn"))
        out.update(pooled_patch_precision=_ratio(tp, tp + fp), pooled_patch_recall=_ratio(tp, tp + fn),
                   pooled_patch_f1=_ratio(2 * tp, 2 * tp + fp + fn))
        ptp = float(df["pixel_tp"].sum())
        pred, gt = float(df["pred_defect_pixels"].sum()), float(df["gt_defect_pixels"].sum())
        out.update(pooled_pixel_iou=_ratio(ptp, pred + gt - ptp), pooled_pixel_dice=_ratio(2 * ptp, pred + gt),
                   pooled_pixel_precision=_ratio(ptp, pred), pooled_pixel_recall=_ratio(ptp, gt))
    for prefix in sorted({c[:-3] for c in df.columns if re.fullmatch(r"window\d*_tp", c)}):
        # window decisions of the supervised detector (e.g. window16_, window32_): finer than 256x256 patches
        tp, fp, fn = (float(df[f"{prefix}_{k}"].sum()) for k in ("tp", "fp", "fn"))
        out.update({f"{prefix}_{m}": float(df[f"{prefix}_{m}"].mean()) for m in ("precision", "recall", "f1", "auroc")
                    if f"{prefix}_{m}" in df})
        out.update({f"pooled_{prefix}_precision": _ratio(tp, tp + fp), f"pooled_{prefix}_recall": _ratio(tp, tp + fn),
                    f"pooled_{prefix}_f1": _ratio(2 * tp, 2 * tp + fp + fn),
                    f"pooled_{prefix}_iou": _ratio(tp, tp + fp + fn)})
    for r in TOLERANCES:
        if f"pixel_tol{r}_tp_pred" in df.columns and len(df):
            pp = _ratio(float(df[f"pixel_tol{r}_tp_pred"].sum()), float(df["pred_defect_pixels"].sum()))
            pr = _ratio(float(df[f"pixel_tol{r}_tp_gt"].sum()), float(df["gt_defect_pixels"].sum()))
            out.update({f"pixel_tol{r}_f1": float(df[f"pixel_tol{r}_f1"].mean()),
                        f"pooled_pixel_tol{r}_precision": pp, f"pooled_pixel_tol{r}_recall": pr,
                        f"pooled_pixel_tol{r}_f1": 2 * pp * pr / (pp + pr) if pp + pr > 0 else float("nan")})
    if "image_pred" in full.columns:
        out.update({f"image_{k}": v for k, v in image_level_metrics(full).items()})
    return out


def image_level_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Strip-level detection over defective (positive) and defect-free (negative) test strips.

    recall = detection rate; specificity = 1 - false-alarm rate; AUROC ranks strips by
    image score, so it does not depend on alpha.
    """
    y = df["is_defective"].astype(bool).to_numpy()
    pred = df["image_pred"].astype(bool).to_numpy()
    tp, fp = int((y & pred).sum()), int((~y & pred).sum())
    tn, fn = int((~y & ~pred).sum()), int((y & ~pred).sum())
    both = 0 < y.sum() < len(y)
    return {
        "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
        "accuracy": _ratio(tp + tn, len(y)), "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "auroc": float(roc_auc_score(y, df["image_score"])) if both else float("nan"),
    }


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
        descriptor_used: str = "GLCM+LBP",
        **extra
    ):
        combined = {
            "sample_id": sample_id,
            "fabric_type": fabric_type,
            "defect_type": defect_type,
            "descriptor": descriptor_used,
            **patch_metrics,
            **pixel_metrics,
            **extra
        }
        self.records.append(combined)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def summary_table(self, by: str = "fabric_type") -> pd.DataFrame:
        df = self.to_dataframe()
        if "is_defective" in df.columns:
            df = df[df["is_defective"].astype(bool)]
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
