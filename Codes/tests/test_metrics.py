import math

import numpy as np
import pandas as pd

from fabric_defect.metrics import EvaluationSuite, evaluate_patch_classification, evaluate_pixel_localization


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
