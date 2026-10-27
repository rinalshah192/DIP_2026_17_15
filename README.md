# Adaptive Fabric Surface Defect Detection and Localization

ECE501 Digital Image Processing, Group 15 (Rinal Shah, Varun Hotani, Aadi Mehta, Pratik Agrawal).

This project detects and localizes defects in woven fabric using **classical texture descriptors only**, with no deep learning. It runs on the [AITEX Fabric Image Database (AFID)](https://www.aitex.es/afid/).

For each fabric, a model of normal texture is built from defect-free images. Every 256×256 window of a test strip is scored against its fabric's model. Flagged windows are then re-examined at 64×64 to produce a pixel mask.

```
strip ─► fabric ROI ─► illumination flattening ─► median ─► CLAHE
      ─► 256×256 windows (stride 128) ─► GLCM (30-D) + multi-scale LBP (28-D)  [+ Gabor (24-D), optional]
      ─► per-fabric Mahalanobis reference ─► τ from held-out defect-free images ─► flagged windows
      ─► 64×64 re-scoring inside flagged windows ─► morphology ─► predicted mask ─► patch + pixel metrics
```

---

## Repository layout

`Codes/` follows the [PyPA src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) and the module split of [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/): data, features, modeling, evaluation, visualization. The code is installed before it is imported, and each layer has its own subpackage.

```
Codes/                              # all source code (course-required folder)
  pyproject.toml                    # installable package (src layout)
  configs/default.toml              # every tunable parameter, with defaults
  configs/experiments/ablation.toml # the Week-6 ablation grid
  scripts/download_afid.sh          # fetches AFID into data/raw
  src/fabric_defect/
    __main__.py                     # `python -m fabric_defect` hook (Python requires it at the root)
    cli/main.py                     # argparse: `run` and `ablate` subcommands
    core/config.py, core/paths.py   # typed config (dataclasses + TOML) and repository paths
    data/afid.py, data/synthetic.py # AFID catalog, official codes, masks; synthetic smoke-test strips
    features/                       # preprocessing, glcm, lbp, gabor, fused descriptor, feature cache
    modeling/                       # Mahalanobis reference, one-class baselines, 64×64 refinement, segmentation
    evaluation/metrics.py           # patch / pixel metrics, per-image and pooled summaries
    pipelines/benchmark.py          # end-to-end run
    pipelines/ablation.py           # runs an ablation grid on shared features
    visualization/plots.py          # inspection figures and charts
  tests/                            # mirrors the package; 72 tests
Results/final/                      # full AFID benchmark with the default config
Results/ablation/                   # one CSV per ablation variant + summary.csv
Results/legacy/                     # mid-semester CSVs and early synthetic figures (kept for reference)
Mid_Sem_Report/, End_Sem_Report/    # course reports
reports/week-N/                     # weekly progress reports (LaTeX + PDF from week 5)
data/raw/                           # AFID (git-ignored)
data/processed/                     # feature cache (git-ignored)
.github/workflows/tests.yml         # CI: tests on every push / PR
```

Imports only point downward. `cli` uses `pipelines`. `pipelines` uses `data`, `features`, `modeling`, `evaluation` and `visualization`. Everything uses `core`.

---

## 1. Setup

You need **Python 3.11 or newer**.

```bash
git clone https://github.com/rinalshah192/DIP_2026_17_15.git
cd DIP_2026_17_15
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e "Codes[dev]"
python -m fabric_defect run --help
```

> **macOS / NumPy 2 note:** if `import cv2` fails with `_ARRAY_API not found`, your system Python has an old OpenCV build. Use the virtual environment above, which installs `opencv-python-headless` built for NumPy 2.

---

## 2. Get the dataset

```bash
bash Codes/scripts/download_afid.sh     # needs curl and 7z (brew install p7zip / apt install p7zip-full)
```

The script downloads the three official archives (about 170 MB) from aitex.es and extracts them into `data/raw/`:

```
data/raw/NODefect_images/   # 140 defect-free strips in 7 per-fabric folders
data/raw/Defect_images/     # 106 defective strips
data/raw/Mask_images/       # <image>_mask.png, or _mask1/_mask2 when a defect has two parts
```

- Images are **256 × 4096** px.
- Filenames follow **`nnnn_ddd_ff.png`**: image id, defect code, fabric code `00`–`06`. Fabrics are reported as `fabric_00` … `fabric_06`.

| Code | Defect | Code | Defect |
|---|---|---|---|
| 000 | none (defect-free) | 022 | cut selvage |
| 002 | broken end | 023 | crease |
| 006 | broken yarn | 025 | warp ball |
| 010 | broken pick | 027 | knot |
| 016 | weft curling | 029 | contamination |
| 019 | fuzzy ball | 030 | nep |
| | | 036 | weft crack |

The pipeline handles these quirks in the published archives:

| File | Quirk | Handling |
|---|---|---|
| `NODefect_images/2608691-202020u/0018_00_01.png` | 2-digit defect code (21st image in that folder) | skipped with a warning |
| `0100_025_08.png` | fabric code 08 does not exist, and there is no mask | skipped with a warning |
| `0044_019_04`, `0097_030_03` | mask split into `_mask1` + `_mask2` | parts merged |
| `0106_010_03` | mask is empty | evaluated; recall-type metrics are NaN |
| `0094_027_05.png` | 256 × 3796 instead of 4096 wide | handled (any width works) |

---

## 3. Run

```bash
# Full benchmark: every defective strip with a mask (105), default config, 8 processes
python -m fabric_defect run --out Results/final --workers 8

# The Week-6 ablation grid: 15 variants on shared, cached features (under a minute once features are cached)
python -m fabric_defect ablate --grid Codes/configs/experiments/ablation.toml --out Results/ablation --workers 8

# Overrides
python -m fabric_defect run --scoring ocsvm --seed 1 --max-test 30
python -m fabric_defect run --config my_experiment.toml --out Results/runs/my_experiment

# Smoke test without AFID (synthetic strips; NOT dataset results)
python -m fabric_defect run --synthetic --out /tmp/smoke
```

**Feature cache.** Feature extraction is the only slow step: about 60 s for all 246 images on 8 processes, or about 160 s including the 64×64 windows. Every feature block (GLCM, LBP, Gabor) is computed once per image and window size. It is stored in `data/processed/` and reused by any run that changes only modelling, scoring or threshold settings. A warm run takes about 4 s. Use `--no-cache` to recompute, or `--cache-dir` to move the cache.

**Exit codes:** `0` success · `1` nothing evaluated · `2` dataset missing or invalid config/grid.

### What a run does

1. **Catalog.** Scan `data/raw`, parse the fabric and defect code from each filename, and merge mask parts.
2. **Split.** Split each fabric's defect-free images **by image** into a fit set (75 %) and a calibration set (25 %), using a seeded split.
3. **Fit.** Build each fabric's reference (mean and regularised covariance) on the fit images, at 256×256 and at 64×64.
4. **Calibrate.** τ = mean + k·std of the reference's scores on the **held-out** calibration images (k = 2.5).
5. **Detect.** Flag 256×256 windows whose score is ≥ τ.
6. **Refine.** Re-score the 64×64 windows that lie inside flagged regions, against the 64×64 model.
7. **Clean up.** Apply morphology to get the predicted mask.
8. **Evaluate and write.** Compute the metrics and write CSVs, `summary.json`, `run_config.json` and figures.

### Outputs

| File | Contents |
|---|---|
| `metrics_summary.csv` | One row per strip: patch TP/FP/TN/FN, accuracy, precision, recall, F1, specificity; pixel IoU, Dice, precision, recall, pixel counts |
| `fabric_summary.csv`, `defect_summary.csv` | Per-fabric and per-defect means plus `n_images` |
| `summary.json` | Overall per-image means **and pooled metrics** (TP/FP/FN summed over all strips first) |
| `run_config.json` | Time, git SHA and dirty flag, data folder, package versions, full resolved config (and the grid for ablations) |
| `figures/` | Inspection collages (raw, pre-processed, deviation, ground truth, coarse mask, refined mask) and charts |

**Metric conventions:**
- A ratio with a zero denominator is **NaN, not 0**. Means skip NaN, and `n_images` says how many strips were averaged.
- *Pooled* metrics weight every pixel or patch equally.
- *Mean* metrics weight every strip equally, so a 50-px nep counts as much as a full-width crease.

---

## 4. Configuration

All parameters live in `Codes/configs/default.toml`, which mirrors `src/fabric_defect/core/config.py`. Unknown keys and invalid values are rejected at start-up.

| Section | Main keys (default) |
|---|---|
| `preprocess` | `median_ksize` 3, `contrast` "clahe" (clip 2.0, 8×8 grid), `illumination_sigma` 50, `selvage_margin` 40 |
| `patch` | `size` 256, `stride` 128, `defect_min_pixels` 25 |
| `glcm` / `lbp` / `gabor` | GLCM: 64 levels, distances 1–3, 4 angles. LBP: (8,1) + (16,2) riu2. Gabor: λ 4/8/16 px, 4 orientations, 1-octave bandwidth |
| `features` | `use` ["glcm", "lbp"] (any of glcm, lbp, gabor) |
| `model` | `scoring` "mahalanobis", `reference` "per_fabric", `k` 2.5, `calibration_fraction` 0.25, `regularization` 1e-4, `ocsvm_nu` 0.1, `iforest_trees` 200 |
| `refine` | `enabled` true, `size` 64, `stride` 32, `k` 2.5 |
| `segment` | `kernel_size` 5, `min_area` 50 |
| top level | `max_test_samples` 0 (= all), `seed` 0 |

**Scoring modes (`model.scoring`):**
- `mahalanobis`: distance to the fabric's defect-free model, thresholded at the held-out τ. This is the default.
- `strip_mahalanobis`: the same distance, robust-z-scored within the strip. It removes a strip-wide offset; high specificity, low recall.
- `relative`: robust z against the other windows of the same strip.
- `hybrid`: 0.6 × relative + 0.4 × robust Mahalanobis. This was the Week-5 default.
- `ocsvm`, `iforest`: One-Class SVM and Isolation Forest baselines. They use the same features, split and threshold rule.

`model.reference = "global"` fits one model and one τ on all fabrics pooled: the *fixed* counterpart to the *adaptive* per-fabric default.

**Ablation grids** are TOML files with a `base` config and `[[variant]]` tables. Each variant has a `name` and `overrides` in the same shape as the config. See `Codes/configs/experiments/ablation.toml`.

---

## 5. Results (AFID, 105 defective strips with masks)

Full tables are in `Results/final/` and `Results/ablation/summary.csv`; discussion is in `reports/week-6/Week_6_Report.pdf`.

| Setting | Patch F1 | Pooled patch F1 | Pixel IoU | Pixel recall |
|---|---|---|---|---|
| Week-5 setting (hybrid, GLCM+LBP, no refinement) | 13.97 % | 14.78 % | 0.30 % | 32.73 % |
| **Default** (Mahalanobis, GLCM+LBP, 64×64 refinement) | **18.14 %** | **22.74 %** | **1.36 %** | **48.39 %** |
| Fixed global reference instead of per-fabric | 5.78 % | 12.03 % | 0.07 % | 22.82 % |
| One-Class SVM baseline | 16.47 % | 23.41 % | 1.30 % | 43.65 % |
| Isolation Forest baseline | 13.96 % | 17.96 % | 0.85 % | 39.39 % |
| Default without refinement | 18.14 % | 22.74 % | 0.70 % | 53.89 % |

Key finding: for 5 of the 7 fabrics, AFID's defect-free and defective strips differ systematically. Normal windows of defective strips score far above the held-out defect-free calibration images. This false-positive floor, not the descriptor, is what limits precision. See the Week 6 report.

Earlier numbers are in `Results/legacy/` and should not be reused:
- the mid-semester 13-image CSVs, whose defect labels came from the old code map
- the synthetic Week-3 figures

---

## 6. Tests and CI

```bash
pytest Codes/tests -q        # 72 tests, about 2 min (includes synthetic end-to-end runs)
```

The tests mirror the package layout. They cover:
- AFID parsing, quirks and mask merging
- ROI and windows
- GLCM, LBP and Gabor (including orientation and wavelength selectivity)
- the feature cache (round trip, parallel extraction, per-scale reuse)
- the Mahalanobis maths and held-out calibration
- strip normalisation, the baselines and refinement
- the NaN and pooled metric rules
- config and grid validation
- end-to-end `run` and `ablate` through the CLI

GitHub Actions runs the suite on Python 3.11 and 3.13.

---

## 7. Contributing

- Work on a branch and open a PR into `main`. CI must pass.
- Use one change per commit with a one-line message in the form `type: summary`, where type is `fix`, `feat`, `refactor`, `perf`, `test`, `docs`, `chore` or `ci`.
- Add a test with every fix.
- Never commit dataset images or the feature cache. `data/` is git-ignored.

## References

- J. Silvestre-Blanes et al., "A public fabric database for defect detection methods and results," *AUTEX Res. J.*, 19(4), 2019.
- R. M. Haralick, K. Shanmugam, I. Dinstein, "Textural features for image classification," *IEEE Trans. SMC*, 1973.
- T. Ojala, M. Pietikäinen, T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE TPAMI*, 2002.
- A. K. Jain, F. Farrokhnia, "Unsupervised texture segmentation using Gabor filters," *Pattern Recognition*, 1991.
- B. Schölkopf et al., "Estimating the support of a high-dimensional distribution," *Neural Computation*, 2001.
- F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation forest," *ICDM*, 2008.
- P. C. Mahalanobis, "On the generalized distance in statistics," 1936.
