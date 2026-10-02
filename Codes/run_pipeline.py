"""
End-to-End Pipeline Runner for Fabric Surface Defect Detection.
ECE501 - Digital Image Processing (Group 15)

Executes:
1. Dataset ingestion / benchmark verification
2. Pre-processing & Patch Extraction
3. GLCM & Multi-Scale LBP Feature Extraction
4. Normal Reference Modeling & Continuous Deviation Mapping
5. Defect Localization & Morphological Refinement
6. Standardized Evaluation Harness & Visual Artifact Generation
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Codes.config import (
    RAW_DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    FABRIC_TYPES,
    STRIP_HEIGHT,
    STRIP_WIDTH,
    PATCH_SIZE,
    PATCH_STRIDE,
    ADAPTIVE_THRESHOLD_K
)
from Codes.dataset import AITEXDataset, generate_synthetic_aitex_benchmark
from Codes.preprocessing import Preprocessor, extract_patches
from Codes.features_glcm import GLCMFeatureExtractor
from Codes.features_lbp import LBPFeatureExtractor
from Codes.reference_model import TextureReferenceModel
from Codes.segmenter import DefectSegmenter
from Codes.evaluation import (
    evaluate_patch_classification,
    evaluate_pixel_localization,
    EvaluationSuite
)


def run_full_pipeline(stride: int = PATCH_STRIDE, max_test_samples: int = 14):
    print("=" * 70)
    print("ECE501: Adaptive Fabric Surface Defect Detection & Localization")
    print("Group 15 - Week 3 Full Pipeline Execution")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset verification
    dataset = AITEXDataset(RAW_DATA_DIR)
    manifest = dataset.scan_catalog()

    if len(manifest) == 0:
        print("\n[+] No raw dataset detected in data/raw/. Generating reproducible benchmark...")
        manifest = generate_synthetic_aitex_benchmark(RAW_DATA_DIR, samples_per_type=2)

    print(f"\n[+] Total catalogued images: {len(manifest)}")
    clean_images = manifest[~manifest["is_defective"]]
    defect_images = manifest[manifest["is_defective"]]
    print(f"    - Defect-free reference strips: {len(clean_images)}")
    print(f"    - Defective test strips:        {len(defect_images)}")

    # 2. Initialize feature extractors & engines
    preprocessor = Preprocessor(use_median=True, use_clahe=True)
    glcm_extractor = GLCMFeatureExtractor()
    lbp_extractor = LBPFeatureExtractor()
    segmenter = DefectSegmenter(method="adaptive", adaptive_k=ADAPTIVE_THRESHOLD_K)
    eval_suite = EvaluationSuite()

    # 3. Fit Reference Models per fabric type
    print("\n[+] Phase 1: Fitting Normal Fabric Reference Models (GLCM + LBP)...")
    reference_models = {}

    for fabric_name in clean_images["fabric_type"].unique():
        fabric_cleans = clean_images[clean_images["fabric_type"] == fabric_name]
        clean_features = []

        for _, row in fabric_cleans.iterrows():
            img = preprocessor.load_image(row["image_path"])
            proc = preprocessor.process(img)["normalized"]
            patch_data = extract_patches(proc, patch_size=PATCH_SIZE, stride=stride)

            glcm_feats = glcm_extractor.extract_batch(patch_data["patches"])
            lbp_feats = lbp_extractor.extract_batch(patch_data["patches"])
            fused = np.hstack([glcm_feats, lbp_feats])
            clean_features.append(fused)

        X_normal = np.vstack(clean_features)
        ref_model = TextureReferenceModel(fabric_type=fabric_name, metric="mahalanobis")
        ref_model.fit(X_normal)
        reference_models[fabric_name] = ref_model
        print(f"    - {fabric_name}: Fitted on {len(X_normal)} patches. Baseline tau = {ref_model.adaptive_threshold:.3f}")

    # 4. Evaluate Test Strips
    print("\n[+] Phase 2: Running Defect Detection & Localization on Test Strips...")
    test_subset = defect_images.head(max_test_samples)
    visualized_count = 0

    for idx, (_, row) in enumerate(test_subset.iterrows()):
        img_path = row["image_path"]
        mask_path = row["mask_path"]
        fabric_name = row["fabric_type"]
        defect_name = row["defect_type"]
        filename = row["filename"]

        ref_model = reference_models.get(fabric_name)
        if ref_model is None:
            continue

        raw_img = preprocessor.load_image(img_path)
        gt_mask = preprocessor.load_mask(mask_path, target_shape=raw_img.shape) if mask_path else np.zeros_like(raw_img)

        # Preprocessing
        proc_dict = preprocessor.process(raw_img)
        norm_img = proc_dict["normalized"]

        # Patch extraction
        patch_dict = extract_patches(norm_img, mask=gt_mask, patch_size=PATCH_SIZE, stride=stride)
        patches = patch_dict["patches"]
        coords = patch_dict["coordinates"]
        patch_labels = patch_dict["labels"]

        # Feature extraction
        glcm_f = glcm_extractor.extract_batch(patches)
        lbp_f = lbp_extractor.extract_batch(patches)
        fused_f = np.hstack([glcm_f, lbp_f])

        # Anomaly scoring & deviation heatmap
        patch_scores = ref_model.score_batch(fused_f)
        heatmap = ref_model.generate_deviation_map(
            patch_scores,
            coordinates=coords,
            strip_shape=raw_img.shape,
            patch_size=PATCH_SIZE
        )

        # Segmentation
        seg_res = segmenter.segment(heatmap, reference_threshold=ref_model.adaptive_threshold)
        pred_mask = seg_res["cleaned_mask"]

        # Patch prediction decisions
        pred_patch_labels = (patch_scores >= ref_model.adaptive_threshold).astype(int)

        # Metrics
        p_metrics = evaluate_patch_classification(patch_labels, pred_patch_labels)
        pix_metrics = evaluate_pixel_localization(pred_mask, gt_mask)

        eval_suite.log_result(
            sample_id=filename,
            fabric_type=fabric_name,
            defect_type=defect_name,
            patch_metrics=p_metrics,
            pixel_metrics=pix_metrics,
            descriptor_used="GLCM+LBP (Fused)"
        )

        # Save Visual Collage for first 3 representative defective strips
        if visualized_count < 3 and np.count_nonzero(gt_mask) > 0:
            _save_visual_inspection_figure(
                sample_id=Path(filename).stem,
                raw_img=raw_img,
                norm_img=norm_img,
                heatmap=heatmap,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                patch_scores=patch_scores,
                threshold=ref_model.adaptive_threshold,
                defect_type=defect_name,
                fabric_type=fabric_name
            )
            visualized_count += 1

    # 5. Output Summary
    results_df = eval_suite.to_dataframe()
    results_csv = RESULTS_DIR / "metrics_summary.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n[+] Saved detailed evaluation metrics to: {results_csv}")

    summary_df = eval_suite.summary_table()
    summary_csv = RESULTS_DIR / "fabric_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 70)
    print("EVALUATION HARNESS BENCHMARK SUMMARY (AITEX AFID)")
    print("=" * 70)
    mean_iou = results_df["pixel_iou"].mean()
    mean_dice = results_df["pixel_dice"].mean()
    mean_patch_f1 = results_df["patch_f1"].mean()
    mean_patch_acc = results_df["patch_accuracy"].mean()
    print(f"Overall Patch Classification Accuracy: {mean_patch_acc * 100:.2f}%")
    print(f"Overall Patch Classification F1-Score: {mean_patch_f1 * 100:.2f}%")
    print(f"Overall Pixel Localization Mean IoU:   {mean_iou * 100:.2f}%")
    print(f"Overall Pixel Localization Mean Dice:  {mean_dice * 100:.2f}%")
    print("=" * 70)

    # Save summary bar plot
    _save_summary_charts(results_df)

    return results_df


def _save_visual_inspection_figure(
    sample_id: str,
    raw_img: np.ndarray,
    norm_img: np.ndarray,
    heatmap: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    patch_scores: np.ndarray,
    threshold: float,
    defect_type: str,
    fabric_type: str
):
    """Generates and saves full 5-panel inspection strip collage."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 12), sharey=True)

    # 1. Raw strip
    axes[0].imshow(raw_img, cmap="gray", aspect="auto")
    axes[0].set_title("1. Raw Fabric Strip\n(4096 x 256)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. CLAHE Normalized
    axes[1].imshow(norm_img, cmap="gray", aspect="auto")
    axes[1].set_title("2. Pre-processed\n(Median + CLAHE)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Anomaly Deviation Heatmap
    im = axes[2].imshow(heatmap, cmap="inferno", aspect="auto")
    axes[2].set_title(f"3. Texture Deviation\n(Mahalanobis D_M)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Ground Truth Mask
    axes[3].imshow(gt_mask, cmap="Blues_r", aspect="auto")
    axes[3].set_title(f"4. Ground Truth Mask\n({defect_type})", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # 5. Predicted Defect Localization Mask
    axes[4].imshow(pred_mask, cmap="Reds", aspect="auto")
    axes[4].set_title("5. Localized Defect\n(tau=%.2f + Morph)" % threshold, fontsize=11, fontweight="bold")
    axes[4].axis("off")

    plt.suptitle(
        f"Defect Localization Inspection - {fabric_type} | Defect: {defect_type}\nSample: {sample_id}",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()

    out_file = FIGURES_DIR / f"inspection_{sample_id}.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved inspection collage: {out_file}")


def _save_summary_charts(df: pd.DataFrame):
    """Generate benchmark metric bar charts."""
    if df.empty:
        return

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Metric 1: Patch F1 and Accuracy by Defect Type
    def_summary = df.groupby("defect_type")[["patch_f1", "patch_accuracy", "pixel_iou"]].mean()
    def_summary.plot(kind="bar", ax=ax[0], colormap="viridis", edgecolor="black", width=0.7)
    ax[0].set_title("Performance Metrics across Defect Types", fontweight="bold")
    ax[0].set_ylabel("Score [0.0 - 1.0]")
    ax[0].set_ylim(0, 1.1)
    ax[0].grid(axis="y", linestyle="--", alpha=0.5)
    ax[0].tick_params(axis="x", rotation=45)

    # Metric 2: Pixel IoU by Fabric Type
    fab_summary = df.groupby("fabric_type")[["pixel_iou", "pixel_dice"]].mean()
    fab_summary.plot(kind="bar", ax=ax[1], colormap="magma", edgecolor="black", width=0.6)
    ax[1].set_title("Pixel Localization (IoU & Dice) by Fabric Weave", fontweight="bold")
    ax[1].set_ylabel("Score [0.0 - 1.0]")
    ax[1].set_ylim(0, 1.1)
    ax[1].grid(axis="y", linestyle="--", alpha=0.5)
    ax[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    chart_file = FIGURES_DIR / "benchmark_performance_charts.png"
    plt.savefig(chart_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved benchmark chart: {chart_file}")


if __name__ == "__main__":
    run_full_pipeline()
