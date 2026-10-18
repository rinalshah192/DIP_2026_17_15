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

from pathlib import Path
import numpy as np
import pandas as pd

from fabric_defect.config import RAW_DATA_DIR, RESULTS_DIR, PipelineConfig
from fabric_defect.afid import AITEXDataset, load_mask
from fabric_defect.synthetic import generate_synthetic_aitex_benchmark
from fabric_defect.preprocessing import Preprocessor, detect_fabric_roi, extract_patches, load_image
from fabric_defect.features import TextureDescriptor
from fabric_defect.model import TextureReferenceModel
from fabric_defect.segmentation import DefectSegmenter
from fabric_defect.viz import save_inspection_figure, save_summary_charts
from fabric_defect.metrics import (
    evaluate_patch_classification,
    evaluate_pixel_localization,
    EvaluationSuite
)


def split_images(paths, calibration_fraction, rng):
    """Image-level split of defect-free images into (fit, calibration) lists.

    Splitting by image (not by patch) keeps overlapping patches of one strip on
    the same side. At least one image always stays in the fit set.
    """
    paths = list(paths)
    if calibration_fraction <= 0 or len(paths) < 2:
        return paths, []
    n_calib = min(len(paths) - 1, max(1, round(len(paths) * calibration_fraction)))
    order = rng.permutation(len(paths))
    calib = [paths[i] for i in order[:n_calib]]
    fit = [paths[i] for i in order[n_calib:]]
    return fit, calib


def run_full_pipeline(
    cfg: PipelineConfig = PipelineConfig(),
    data_dir: Path = RAW_DATA_DIR,
    out_dir: Path = RESULTS_DIR,
    synthetic: bool = False
) -> pd.DataFrame:
    print("=" * 70)
    print("ECE501: Adaptive Fabric Surface Defect Detection & Localization")
    print("=" * 70)

    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    if synthetic:
        print("\n[+] SYNTHETIC smoke run: results are not AFID results.")
        manifest = generate_synthetic_aitex_benchmark(Path(data_dir), samples_per_type=2)
    else:
        manifest = AITEXDataset(data_dir).scan_catalog()

    print(f"\n[+] Total catalogued images: {len(manifest)}")
    clean_images = manifest[~manifest["is_defective"]]
    defect_images = manifest[manifest["is_defective"]]
    print(f"    - Defect-free reference strips: {len(clean_images)}")
    print(f"    - Defective test strips:        {len(defect_images)}")

    # 2. Engines
    preprocessor = Preprocessor(cfg.preprocess)
    descriptor = TextureDescriptor(cfg.glcm, cfg.lbp)
    segmenter = DefectSegmenter(cfg.segment, method="adaptive", adaptive_k=cfg.model.k)
    eval_suite = EvaluationSuite()
    psize, stride = cfg.patch.size, cfg.patch.stride

    def roi(img):
        return detect_fabric_roi(img, margin=cfg.preprocess.selvage_margin, min_width=psize)

    # 3. Fit reference models per fabric type (on the genuine fabric ROI)
    print("\n[+] Phase 1: Fitting Normal Fabric Reference Models (GLCM + LBP)...")
    reference_models = {}

    def image_features(paths):
        feats = []
        for path in paths:
            img = load_image(path)
            c0, c1 = roi(img)
            proc = preprocessor.process(img[:, c0:c1])["normalized"]
            patches = extract_patches(proc, patch_size=psize, stride=stride)["patches"]
            if len(patches):
                feats.append(descriptor.extract_batch(patches))
        return np.vstack(feats) if feats else None

    rng = np.random.default_rng(cfg.seed)
    for fabric_name in sorted(clean_images["fabric_type"].unique()):
        paths = clean_images.loc[clean_images["fabric_type"] == fabric_name, "image_path"].tolist()
        fit_paths, calib_paths = split_images(paths, cfg.model.calibration_fraction, rng)

        X_fit = image_features(fit_paths)
        if X_fit is None:
            continue
        X_calib = image_features(calib_paths)

        ref_model = TextureReferenceModel(fabric_type=fabric_name, cfg=cfg.model).fit(X_fit, X_calib)
        reference_models[fabric_name] = ref_model
        source = f"{len(calib_paths)} held-out images" if ref_model.calibrated_out_of_sample else "in-sample (too few images)"
        print(f"    - {fabric_name}: fit on {len(fit_paths)} images / {len(X_fit)} patches; "
              f"tau = {ref_model.adaptive_threshold:.3f} from {source}")

    # 4. Evaluate test strips, stratified across fabric types
    print("\n[+] Phase 2: Running Defect Detection & Localization on Test Strips...")
    unique_fabrics = defect_images["fabric_type"].unique()
    samples_per_type = max(1, cfg.max_test_samples // max(1, len(unique_fabrics)))
    # groupby().head keeps the fabric_type column (groupby().apply drops it on pandas >= 3)
    test_subset = defect_images.groupby("fabric_type").head(samples_per_type).head(cfg.max_test_samples)
    print(f"    - Testing {len(test_subset)} stratified test strips across {len(unique_fabrics)} fabric types.")
    visualized_count = 0

    for _, row in test_subset.iterrows():
        mask_paths = row["mask_paths"]
        fabric_name = row["fabric_type"]
        defect_name = row["defect_type"]
        filename = row["filename"]

        ref_model = reference_models.get(fabric_name)
        if ref_model is None:
            continue
        if not mask_paths:
            # An empty ground truth would silently score every detection as a false positive
            print(f"    ! Skipping {filename}: no ground-truth mask found")
            continue

        raw_img = load_image(row["image_path"])
        gt_mask = load_mask(mask_paths, raw_img.shape)
        if not gt_mask.any():
            print(f"    ! {filename}: ground-truth mask is empty; recall-type metrics will be NaN")

        # ROI crop (drops conveyor padding and selvage edge), then preprocessing
        c0, c1 = roi(raw_img)
        norm_crop = preprocessor.process(raw_img[:, c0:c1])["normalized"]
        norm_img = np.full_like(raw_img, 255)
        norm_img[:, c0:c1] = norm_crop

        patch_dict = extract_patches(
            norm_crop, mask=gt_mask[:, c0:c1], patch_size=psize, stride=stride,
            offset_x=c0, defect_min_pixels=cfg.patch.defect_min_pixels
        )
        patches = patch_dict["patches"]
        if len(patches) == 0:
            continue

        # Scoring, deviation heatmap, segmentation
        patch_scores, adaptive_thresh = ref_model.score_strip(descriptor.extract_batch(patches))
        heatmap = ref_model.generate_deviation_map(
            patch_scores,
            coordinates=patch_dict["coordinates"],
            strip_shape=raw_img.shape,
            patch_size=psize
        )
        pred_mask = segmenter.segment(heatmap, reference_threshold=adaptive_thresh)["cleaned_mask"]
        pred_patch_labels = (patch_scores >= adaptive_thresh).astype(int)

        eval_suite.log_result(
            sample_id=filename,
            fabric_type=fabric_name,
            defect_type=defect_name,
            patch_metrics=evaluate_patch_classification(patch_dict["labels"], pred_patch_labels),
            pixel_metrics=evaluate_pixel_localization(pred_mask, gt_mask),
            descriptor_used=f"GLCM+LBP ({cfg.model.scoring})"
        )

        # Visual collage for the first 3 defective strips
        if visualized_count < 3 and np.count_nonzero(gt_mask) > 0:
            fig_path = save_inspection_figure(
                out_file=figures_dir / f"inspection_{Path(filename).stem}.png",
                sample_id=Path(filename).stem,
                raw_img=raw_img,
                norm_img=norm_img,
                heatmap=heatmap,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                threshold=adaptive_thresh,
                defect_type=defect_name,
                fabric_type=fabric_name
            )
            print(f"[+] Saved inspection collage: {fig_path}")
            visualized_count += 1

    # 5. Outputs
    results_df = eval_suite.to_dataframe()
    results_csv = out_dir / "metrics_summary.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n[+] Saved detailed evaluation metrics to: {results_csv}")
    eval_suite.summary_table().to_csv(out_dir / "fabric_summary.csv", index=False)

    if results_df.empty:
        print("[!] No test strips were evaluated.")
        return results_df

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY" + (" (SYNTHETIC)" if synthetic else " (AFID)"))
    print("=" * 70)
    print(f"Overall Patch Classification Accuracy: {results_df['patch_accuracy'].mean() * 100:.2f}%")
    print(f"Overall Patch Classification F1-Score: {results_df['patch_f1'].mean() * 100:.2f}%")
    print(f"Overall Pixel Localization Mean IoU:   {results_df['pixel_iou'].mean() * 100:.2f}%")
    print(f"Overall Pixel Localization Mean Dice:  {results_df['pixel_dice'].mean() * 100:.2f}%")
    print("=" * 70)

    chart = save_summary_charts(results_df, figures_dir / "benchmark_performance_charts.png")
    print(f"[+] Saved benchmark chart: {chart}")
    return results_df
