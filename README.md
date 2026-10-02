# Adaptive Fabric Surface Defect Detection and Localization using Classical Texture Descriptors

[![Course](https://img.shields.io/badge/Course-ECE501%20Digital%20Image%20Processing-blue.svg)](https://ahduni.edu.in)
[![Group](https://img.shields.io/badge/Group-15-green.svg)]()
[![Status](https://img.shields.io/badge/Week%203-Completed-brightgreen.svg)]()

Academic repository for the **ECE501 Digital Image Processing (DIP)** course project on automated, adaptive surface inspection and defect localization for industrial woven fabrics using classical texture descriptors.

---

## 👥 Group Members (Group 15)
* **Rinal Shah**
* **Varun Hotani**
* **Aadi Mehta**
* **Pratik Agrawal**

---

## 📌 Project Overview
Fabric defect detection is a vital quality control step in the textile manufacturing process. Manual inspection suffers from human fatigue, slow throughput, and inconsistency. This project develops an end-to-end computer vision pipeline using **classical Digital Image Processing (DIP) techniques** to detect and localize surface anomalies on industrial fabrics without relying on heavy deep neural networks.

The system evaluates:
* **Pre-processing:** Illumination gradient compensation via CLAHE and edge-preserving median denoising.
* **Classical Texture Descriptors:** Gray-Level Co-occurrence Matrix (GLCM) Haralick statistics and Multi-Scale Rotation-Invariant Uniform Local Binary Patterns ($LBP_{P, R}^{riu2}$).
* **Reference Anomaly Modeling:** Statistical baseline modeling of defect-free weaves and continuous Mahalanobis distance deviation mapping.
* **Localization & Morphology:** Adaptive statistical thresholding ($\tau = \mu + k\sigma$) and morphological cleanup.
* **Standardized Evaluation:** Quantitative pixel-level (IoU, Dice) and patch-level (Accuracy, Precision, Recall, F1) benchmarks against ground truth.

---

## 📂 Repository Directory Structure

```
DIP_2026_FabricSurfaceDefectDetection_15/
├── Codes/                                 # Implementation source code
│   ├── __init__.py
│   ├── config.py                          # Global configuration (patch size, GLCM & LBP parameters)
│   ├── dataset.py                         # AITEX AFID data handler and benchmark generator
│   ├── preprocessing.py                   # Median/Gaussian denoising, CLAHE, patch extraction
│   ├── features_glcm.py                   # GLCM Haralick feature extractor
│   ├── features_lbp.py                    # Multi-scale uniform rotation-invariant LBP extractor
│   ├── reference_model.py                 # Normal texture reference distribution & deviation mapper
│   ├── segmenter.py                       # Adaptive thresholding and morphological filtering
│   ├── evaluation.py                      # Patch-level & pixel-level evaluation harness
│   └── run_pipeline.py                    # End-to-end pipeline runner
│
├── Results/                               # Benchmark outputs and visual inspection artifacts
│   ├── figures/                           # Visual strip inspection collages and performance plots
│   ├── metrics_summary.csv                # Sample-by-sample quantitative results
│   └── fabric_summary.csv                 # Per-fabric aggregated summary table
│
├── reports/                               # Weekly academic progress reports
│   ├── week-1/
│   │   └── Week 1 Project Progress Report.pdf
│   ├── week-2/
│   │   └── Week_2_Project_Progress_Report.pdf
│   └── week-3/
│       ├── README.md                      # Week 3 milestone summary
│       ├── Week_3_Project_Progress_Report.md  # Week 3 report markdown source
│       └── Week_3_Project_Progress_Report.pdf # Official Week 3 submission report
│
├── requirements.txt                       # Python dependencies
└── README.md                              # Main project documentation
```

---

## 🚀 Quick Start & Reproducibility

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/rinalshah192/DIP_2026_FabricSurfaceDefectDetection_15.git
cd DIP_2026_FabricSurfaceDefectDetection_15
pip install -r requirements.txt
```

### 2. Run Full Pipeline
To execute the end-to-end pipeline (data verification, preprocessing, GLCM & LBP extraction, normal modeling, defect localization, and evaluation):
```bash
python Codes/run_pipeline.py
```
Outputs and plots will be automatically exported to `Results/` and `Results/figures/`.

---

## 📊 Summary of Week 3 Benchmark Results

Evaluated across the 7 woven fabric structures and industrial defect categories:

| Evaluation Metric | Score | Note |
| :--- | :---: | :--- |
| **Patch Classification Accuracy** | **95.39%** | High true negative and true positive classification |
| **Patch Classification F1-Score** | **80.87%** | Balanced precision and recall across rare anomaly classes |
| **Pixel Localization Recall** | **100.00%** | Zero missed defects; all flaws captured inside flagged regions |
| **Pixel Localization Mean IoU** | **1.99%** | Classical $256 \times 256$ tile envelope vs hairline thread flaws |

---

## 🗓️ Weekly Milestones
* **Week 1:** Problem definition, literature review on GLCM, LBP, and Gabor filters, repository establishment.
* **Week 2:** Finalization of the AITEX Fabric Image Database (AFID), $256 \times 256$ patch tiling strategy, median denoising, and CLAHE pre-processing module.
* **Week 3 (Current):** GLCM Haralick and multi-scale LBP descriptors, statistical reference modeling, continuous 2D deviation heatmaps, adaptive thresholding, and standardized evaluation harness.
* **Week 4 (Planned):** Multi-orientation Gabor filter banks, hierarchical multi-scale sub-patch refinement ($64 \times 64$ nested windows to boost pixel IoU), and supervised anomaly classifiers (One-Class SVM / Isolation Forest).
