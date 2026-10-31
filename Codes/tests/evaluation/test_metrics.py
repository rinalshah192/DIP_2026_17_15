import math

import numpy as np
import pandas as pd

from fabric_defect.evaluation.metrics import (
    EvaluationSuite,
    evaluate_patch_classification,
    evaluate_pixel_localization,
    summarize,
)


def test_patch_metrics_known_counts():
    m = evaluate_patch_classification([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert (m["patch_tp"], m["patch_fp"], m["patch_tn"], m["patch_fn"]) == (1, 1, 2, 1)
    assert m["patch_precision"] == 0.5 and m["patch_recall"] == 0.5
    assert m["patch_f1"] == 0.5 and m["patch_accuracy"] == 0.6


def test_undefined_patch_metrics_are_nan_not_zero():
    m = evaluate_patch_classification([0, 0], [0, 0])
    assert math.isnan(m["patch_precision"]) and math.isnan(m["patch_recall"]) and math.isnan(m["patch_f1"])
    missed = evaluate_patch_classification([1, 0], [0, 0])
    assert math.isnan(missed["patch_precision"]) and missed["patch_f1"] == 0.0  # a real miss stays 0


def test_pixel_metrics_and_empty_ground_truth():
    gt = np.zeros((10, 10), np.uint8)
    gt[:5] = 255
    pred = np.zeros_like(gt)
    pred[:, :5] = 255
    m = evaluate_pixel_localization(pred, gt)
    assert m["pixel_iou"] == 25 / 75 and m["pixel_dice"] == 0.5
    empty = evaluate_pixel_localization(pred, np.zeros_like(gt))
    assert math.isnan(empty["pixel_recall"]) and empty["pixel_iou"] == 0.0
    both = evaluate_pixel_localization(np.zeros_like(gt), np.zeros_like(gt))
    assert math.isnan(both["pixel_iou"]) and math.isnan(both["pixel_dice"])


def test_summary_skips_nan_and_counts_images():
    suite = EvaluationSuite()
    for f1 in (0.5, float("nan")):
        suite.log_result("s", "fabric_00", "knot",
                         {"patch_accuracy": 1.0, "patch_precision": f1, "patch_recall": f1, "patch_f1": f1},
                         {"pixel_iou": 0.1, "pixel_dice": 0.2, "pixel_precision": 0.1, "pixel_recall": 0.3})
    row = suite.summary_table().iloc[0]
    assert row["n_images"] == 2 and row["patch_f1"] == 0.5
    assert isinstance(suite.to_dataframe(), pd.DataFrame)


def test_pooled_metrics_weight_pixels_not_images():
    big_gt = np.zeros((10, 100), np.uint8)
    big_gt[:, :50] = 255
    big_pred = big_gt.copy()  # perfect on a 500-px defect
    small_gt = np.zeros((10, 100), np.uint8)
    small_gt[0, 0:5] = 255
    small_pred = np.zeros_like(small_gt)
    small_pred[5, 50:95] = 255  # misses a 5-px defect, 45 false pixels
    rows = []
    for pred, gt in ((big_pred, big_gt), (small_pred, small_gt)):
        rows.append({**evaluate_patch_classification([1, 0], [1, 0]), **evaluate_pixel_localization(pred, gt)})
    s = summarize(pd.DataFrame(rows))
    assert s["n_images"] == 2
    assert s["pixel_iou"] == 0.5  # mean of 1.0 and 0.0
    assert s["pooled_pixel_iou"] == 500 / (500 + 5 + 45)
    assert s["pooled_patch_f1"] == 1.0


def test_image_level_metrics_and_defective_only_summaries():
    from fabric_defect.evaluation.metrics import image_level_metrics
    rows = []
    for defective, pred, score in ((True, True, 3.0), (True, False, 0.8), (False, False, 0.5), (False, True, 2.0)):
        rows.append({**evaluate_patch_classification([1, 0], [1, 0]),
                     **evaluate_pixel_localization(np.full((2, 2), 255 if defective else 0, np.uint8),
                                                   np.full((2, 2), 255 if defective else 0, np.uint8)),
                     "is_defective": defective, "image_pred": pred, "image_score": score})
    df = pd.DataFrame(rows)
    m = image_level_metrics(df)
    assert (m["n_pos"], m["n_neg"]) == (2, 2)
    assert m["recall"] == 0.5 and m["specificity"] == 0.5 and m["auroc"] == 0.75
    s = summarize(df)
    assert s["n_images"] == 2 and s["image_auroc"] == 0.75  # patch/pixel over defective strips only
