# Week 3 Progress Deliverables

This directory contains the Week 3 project progress documentation for the ECE501 Digital Image Processing (DIP) project: **Adaptive Fabric Surface Defect Detection and Localization using Classical Texture Descriptors** (Group 15).

## Files in this Directory:
* **`Week_3_Project_Progress_Report.pdf`**: The official submission-ready progress report for Week 3.
* **`Week_3_Project_Progress_Report.md`**: Complete Markdown source of the Week 3 progress report, including mathematical formulations, experimental benchmark tables, challenge resolutions, and Week 4 plan.

## Summary of Week 3 Milestones Achieved:
1. **GLCM Haralick Feature Extraction:** Implemented in [`Codes/features_glcm.py`](../../Codes/features_glcm.py) with 5 Haralick statistics across multiple distances and orientations.
2. **Multi-Scale Uniform LBP:** Implemented in [`Codes/features_lbp.py`](../../Codes/features_lbp.py) with $P=8, R=1$ and $P=16, R=2$ rotation-invariant uniform histograms.
3. **Normal Fabric Statistical Reference Modeling:** Implemented in [`Codes/reference_model.py`](../../Codes/reference_model.py) using regularized Mahalanobis distance scoring.
4. **Adaptive Defect Localization:** Implemented in [`Codes/segmenter.py`](../../Codes/segmenter.py) combining $\tau = \mu + k\sigma$ statistical thresholding with morphological opening, closing, and small blob suppression.
5. **Standardized Evaluation Harness:** Implemented in [`Codes/evaluation.py`](../../Codes/evaluation.py) and [`Codes/run_pipeline.py`](../../Codes/run_pipeline.py), generating quantitative benchmark metrics in [`Results/metrics_summary.csv`](../../Results/metrics_summary.csv) and inspection collages in [`Results/figures/`](../../Results/figures/).
