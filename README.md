# Adaptive Fabric Surface Defect Detection and Localization

ECE501 Digital Image Processing, Group 15 (Rinal Shah, Varun Hotani, Aadi Mehta, Pratik Agrawal).

This project detects and localizes defects in woven fabric using **classical texture descriptors only**, with no deep learning. It runs on the [AITEX Fabric Image Database (AFID)](https://www.aitex.es/afid/). For each fabric, a model of normal texture is built from defect-free images. Each test strip is then scored patch by patch against that model, and patches that deviate too much are flagged and turned into a defect mask.

```
strip ─► fabric ROI ─► illumination flattening ─► median ─► CLAHE ─► 256×256 patches (stride 128)
      ─► GLCM (30-D) + multi-scale LBP (28-D) ─► per-fabric reference model ─► deviation score
      ─► threshold τ ─► deviation heatmap ─► morphology ─► predicted mask ─► patch + pixel metrics
```

---

## Repository layout

```
Codes/                      # all source code (course-required folder)
  pyproject.toml            # makes the code an installable package
  configs/default.toml      # every tunable parameter, with defaults
  fabric_defect/            # the package
    cli.py, __main__.py     #   command line: python -m fabric_defect run ...
    config.py               #   typed, validated configuration (dataclasses + TOML)
    afid.py                 #   AFID catalog, official defect codes, mask loading
    synthetic.py            #   synthetic strips for smoke tests only
    preprocessing.py        #   ROI, illumination flattening, filtering, CLAHE, patching
    features.py             #   GLCM + LBP texture descriptors
    model.py                #   reference model, Mahalanobis / robust hybrid scoring, τ
    segmentation.py         #   thresholding + morphological clean-up
    metrics.py              #   patch- and pixel-level metrics
    pipeline.py             #   end-to-end orchestration and result writing
    viz.py                  #   inspection figures and summary charts
  tests/                    # pytest suite (42 tests)
Results/                    # metrics CSVs and figures from runs (course-required folder)
Mid_Sem_Report/             # mid-semester IEEE report (LaTeX + PDF)
End_Sem_Report/             # end-semester report (to come)
reports/week-N/             # weekly progress reports
data/raw/                   # put AFID here (git-ignored, not committed)
.github/workflows/tests.yml # CI: runs the tests on every push / PR
```

---

## 1. Setup

You need **Python 3.11 or newer** (the config loader uses `tomllib`).

```bash
git clone https://github.com/rinalshah192/DIP_2026_17_15.git
cd DIP_2026_17_15
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e "Codes[dev]"
```

`pip install -e "Codes[dev]"` installs the package in editable mode, so code edits apply immediately. It also installs the `fabric-defect` command and pytest. `pip install -r requirements.txt` works too, but then you must run commands from inside `Codes/`.

Check the install:

```bash
python -m fabric_defect run --help
```

> **macOS / NumPy 2 note:** if `import cv2` fails with `_ARRAY_API not found`, your system Python has an old OpenCV build. Use the virtual environment above, which installs `opencv-python-headless` built for NumPy 2.

---

## 2. Get the dataset

The images are not in the repo (about 200 MB). Download AFID from <https://www.aitex.es/afid/> and extract it into `data/raw/` so that you have:

```
data/raw/
  NODefect_images/        # 140 defect-free strips, in per-fabric sub-folders
  Defect_images/          # 105 defective strips
  Mask_images/            # ground-truth masks: <image>_mask.png (sometimes _mask1, _mask2, ...)
```

AFID facts that the code relies on:

- Every image is **256 × 4096** px: 256 rows along the fabric's travel direction, 4096 columns across its width.
- Filenames follow **`nnnn_ddd_ff.png`**: image id, defect code, fabric code (`00`–`06`). Fabrics are reported as `fabric_00` … `fabric_06`.
- Defect codes follow the AFID paper:

| Code | Defect | Code | Defect |
|---|---|---|---|
| 000 | none (defect-free) | 022 | cut selvage |
| 002 | broken end | 023 | crease |
| 006 | broken yarn | 025 | warp ball |
| 010 | broken pick | 027 | knot |
| 016 | weft curling | 029 | contamination |
| 019 | fuzzy ball | 030 | nep |
| | | 036 | weft crack |

If a defective image has several mask files, they are merged. If it has no mask, it is **skipped with a warning** and never scored against an empty ground truth.

---

## 3. Run the pipeline

```bash
# Full run on AFID in data/raw, results written to Results/
python -m fabric_defect run

# Same thing via the installed command
fabric-defect run

# Choose the data and output folders
python -m fabric_defect run --data-dir /path/to/afid --out Results/runs/exp1

# Override settings from the command line
python -m fabric_defect run --scoring mahalanobis --seed 1 --max-test 30

# Or from a config file (see section 4)
python -m fabric_defect run --config my_experiment.toml --out Results/runs/my_experiment

# More or less log output
python -m fabric_defect run --log-level DEBUG
```

**Smoke test without AFID.** This generates synthetic woven strips with injected defects and runs the whole pipeline on them (about 45 s):

```bash
python -m fabric_defect run --synthetic --out /tmp/smoke
```

Synthetic results are **not AFID results**. They are labelled `SYNTHETIC` in the log and recorded as `"synthetic": true` in `run_config.json`. Never put them in a report as dataset numbers.

**Exit codes:** `0` success · `1` no strip could be evaluated · `2` dataset missing or invalid config.

### What a run does

1. **Catalog.** Scans `data/raw`, parses fabric and defect from each filename, and finds the masks. If the folder is missing, the run stops with an error rather than falling back to synthetic data.
2. **Fit.** For each fabric, the defect-free images are split **by image** into a fit set (75 %) and a calibration set (25 %), using a seeded split. GLCM + LBP descriptors from the fit set give the mean μ and the regularised covariance Σ.
3. **Calibrate.** τ = mean + k·std of the scores on the **held-out** calibration images, with k = 2.5.
4. **Evaluate.** Each defective strip (14 by default, stratified by fabric) goes through ROI cropping, pre-processing, patching, descriptors, scoring, heatmap, threshold, morphology, and finally the metrics.
5. **Write** the outputs below.

### Outputs (in `--out`, default `Results/`)

| File | Contents |
|---|---|
| `metrics_summary.csv` | One row per test image: patch TP/FP/TN/FN, accuracy, precision, recall, F1, specificity; pixel IoU, Dice, precision, recall, GT and predicted pixel counts |
| `fabric_summary.csv` | Per-fabric means plus `n_images` |
| `run_config.json` | Time, git commit SHA and dirty flag, data folder, synthetic flag, Python and package versions, full resolved config |
| `figures/inspection_<image>.png` | Raw, pre-processed, deviation, ground truth and prediction for the first 3 test strips |
| `figures/benchmark_performance_charts.png` | Mean metrics per defect type and per fabric |

**Metric convention:** a ratio with a zero denominator is reported as **NaN, not 0**. For example, recall is NaN when the mask is empty, and precision is NaN when nothing was predicted. Means skip NaN and `n_images` says how many images were averaged. A real miss (defect present, none found) still counts as 0.

---

## 4. Configuration

All parameters live in `Codes/configs/default.toml`, which mirrors `fabric_defect/config.py`. To experiment, copy it, keep only the keys you want to change, and pass it with `--config`:

```toml
# my_experiment.toml
[patch]
stride = 256            # non-overlapping patches

[model]
scoring = "mahalanobis" # hybrid | relative | mahalanobis
k = 3.0
```

| Section | Main keys (default) |
|---|---|
| `preprocess` | `median_ksize` 3, `gaussian_ksize` 0 (off), `contrast` "clahe", `clahe_clip_limit` 2.0, `clahe_grid` [8, 8], `illumination_sigma` 50 (0 = off), `selvage_margin` 40 |
| `patch` | `size` 256, `stride` 128, `defect_min_pixels` 25 (a patch is defective if its mask has more pixels than this) |
| `glcm` | `distances` [1, 2, 3], `angles_deg` [0, 45, 90, 135], `levels` 64, 5 Haralick properties |
| `lbp` | `scales` [[8, 1], [16, 2]] (uniform, rotation-invariant) |
| `model` | `scoring` "hybrid", `regularization` 1e-4, `k` 2.5, `calibration_fraction` 0.25 |
| `segment` | `kernel_size` 5, `min_area` 50 |
| top level | `max_test_samples` 14, `seed` 0 |

Unknown keys and invalid values are rejected at start-up. Examples: an even median kernel, a stride larger than the patch, an unknown scoring mode.

**Scoring modes:**
- `mahalanobis`: distance of each patch to its fabric's defect-free model, thresholded at the calibrated τ.
- `relative`: robust (median/MAD) z-score of each patch against the other patches **of the same strip**, thresholded at k.
- `hybrid` (default): 0.6 × relative + 0.4 × robust-normalised Mahalanobis, thresholded at k. Because it compares a strip with itself, it can miss defects that span the strip's full width, such as some broken picks. Use `mahalanobis` for those.

---

## 5. Use it from Python

```python
from pathlib import Path
from fabric_defect.config import PipelineConfig
from fabric_defect.pipeline import run_full_pipeline

cfg = PipelineConfig.load("Codes/configs/default.toml")
results = run_full_pipeline(cfg, data_dir=Path("data/raw"), out_dir=Path("Results/runs/notebook"))
print(results[["sample_id", "defect_type", "patch_f1", "pixel_iou"]])
```

Each stage can also be used on its own, for example in a notebook:

```python
from fabric_defect.preprocessing import Preprocessor, extract_patches, load_image
from fabric_defect.features import TextureDescriptor

img = load_image("data/raw/Defect_images/0001_002_00.png")
patches = extract_patches(Preprocessor().process(img)["normalized"])["patches"]
X = TextureDescriptor().extract_batch(patches)   # (n_patches, 58)
```

---

## 6. Tests and CI

```bash
pytest Codes/tests -q        # 42 tests, about 30–45 s
```

The suite covers AFID filename parsing and defect codes, multi-part mask merging, ROI and patch extraction, the descriptor dimensions, the Mahalanobis score (against the explicit formula), held-out calibration, the NaN metric rules, config validation, and an end-to-end synthetic run through the CLI. GitHub Actions (`.github/workflows/tests.yml`) runs it on Python 3.11 and 3.13 for every push and pull request.

---

## 7. Status of the results

- `Results/sample_level_benchmark_13.csv` and `Results/metrics_summary.csv` come from the **mid-semester 13-image AFID run**: patch F1 27.13 %, pixel IoU 0.36 %. Pixel localization is still very coarse.
- That run used the old code, whose defect names were wrong. The `defect_type` column in those CSVs is therefore wrong. Use the official codes table above: for example, `0001_002_00` is a broken end, not a broken pick.
- Under the NaN convention, the same run gives patch recall 49.09 % (n = 11) and pixel recall 47.20 % (n = 12), instead of 41.54 % / 43.57 % over 13 images.
- The `Results/figures/inspection_Fabric01_*` images and the Week 3 results table were produced on **synthetic** data.
- Next step: re-run AFID on the current code and commit `run_config.json` alongside the CSVs.

Details are in `reports/week-5/Week_5_Report.pdf`.

---

## 8. Contributing

- Work on a branch and open a PR into `main`. CI must pass.
- Use one change per commit with a one-line message in the form `type: summary`, where type is `fix`, `feat`, `refactor`, `perf`, `test`, `docs`, `chore` or `ci`.
- Add a test with every fix.
- Never commit dataset images. `data/` is git-ignored.

## References

- J. Silvestre-Blanes et al., "A public fabric database for defect detection methods and results," *AUTEX Res. J.*, 19(4), 2019.
- R. M. Haralick, K. Shanmugam, I. Dinstein, "Textural features for image classification," *IEEE Trans. SMC*, 1973.
- T. Ojala, M. Pietikäinen, T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE TPAMI*, 2002.
- P. C. Mahalanobis, "On the generalized distance in statistics," 1936.
- P. J. Rousseeuw, C. Croux, "Alternatives to the median absolute deviation," *JASA*, 1993.
