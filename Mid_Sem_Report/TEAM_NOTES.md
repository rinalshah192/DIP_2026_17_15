# Mid-Sem Report: team notes (do not submit this file)

Deadline: Sun 11 Oct 2026, 5:00 PM IST. Report + PPT via Google Classroom (POC) **and** GitHub. Presentation 12 Oct, 8 min (1 min script question / 5 min talk / 2 min Q&A).

## A. Outline and writing questions (answer in your own words)

| Section | Write about | Questions to answer |
|---|---|---|
| Abstract | whole report | What problem, what classical method, what scope (13 images), which numbers, what is still weak (pixel IoU 0.36 %), what next? |
| Introduction | why + data + objective | Why is texture variation a problem for fabric inspection? What does AFID contain (245 / 7 / 140+105 / 12 / 4096x256)? Which papers back GLCM [haralick1973] and LBP [ojala2002]? What is the one-sentence objective? What is *not* done yet? |
| Methodology | pipeline + equations | Why median + CLAHE? Why 256x256 patches? What do GLCM and LBP each capture? Why a separate reference per fabric? What does Mahalanobis distance add over Euclidean? How is tau set, on which images? What do opening/closing/min-area do? How are patch labels derived from masks? |
| Results | Table I | Which images, which averaging? Which fabrics are best/worst, with *n*? Why is pixel accuracy ~80 % not a good sign here? |
| Discussions | interpretation | Why are pixel precision/IoU ~0.4 %? What could explain fabric IDs 04 and 06 scoring 0 (mask pairing, GT content, zero-denominator rule)? What does 256x256 do to thin defects? What is untested? |
| Conclusion | summary + next | What does the pipeline demonstrably do? Limits? Next: 64x64 refinement, fixed-vs-adaptive under identical splits, GLCM/LBP/hybrid, larger set with normals, per-image masks, reliability score. |

Rough budget for a 2-page layout (adjust to the layout): abstract ~110 words, intro ~170, method ~260 (plus 3 equations), results ~120 (plus Table I, which spans both columns), discussion ~170, conclusion ~80. Table I and 4 references take a large share of page 2; trim prose, never limitations.

Use a `\cite{}` for GLCM/LBP/AFID where you introduce them. Do not cite these papers as evidence for *our* numbers.

## B. Files
`main.tex` (IEEEtran, compiles with `tectonic main.tex`; verified it builds and BibTeX resolves). The page count currently reflects placeholders only, so **two-page compliance is not yet established**. Re-check after writing.

## C. References
`references.bib`: Haralick 1973, Ojala 2002, Silvestre-Blanes 2019 verified against Crossref (title, volume, issue, pages, year); AFID page text verified (245/7/140/105, 4096x256, filename pattern, defect codes). Not included: Mahalanobis (1936) and Zuiderveld CLAHE (1994, *Graphics Gems IV*) were not verified here; verify before adding. Week-1 refs (Silvestre survey, Asha, Tajeripour, Kim) not re-verified.

## D. Open evidence issues
1. **Upgraded code is not in the repo.** Committed `Codes/` is the older version: wrong defect-code map (e.g. 002 -> broken_pick), fabric inferred as `last_number % 7` (so `ff=00` -> Fabric01 by index only), missing mask -> silent all-zero GT, synthetic-data fallback, test set = `defective.head(14)`. Committed `Results/metrics_summary.csv` is **synthetic** (95 %-range numbers). The 13-image benchmark cannot be reproduced from the commit. Push the upgraded code + the real-run CSV, then re-verify every setting in Methodology.
2. Parameters in `config.py` match your list (256/128, median 3, CLAHE 2.0 8x8, GLCM 64 lv / d 1-3 / 4 angles = 30, LBP (8,1)+(16,2) = 28, D = 58, k = 2.5, kernel 5, min area 50). Committed regularisation is `Sigma + 1e-4*(tr/D)*I` (relative, not plain 1e-4), pseudoinverse; tau = mean + k*std of scores on the **same** reference patches (in-sample).
3. Splits: committed runner fits references on *all* normal images and tests on defective images only; image-level disjoint, but there is no validation set and tau is calibrated in-sample. Number of reference images not stated for the real run.
4. Patch GT rule (committed): defective if > 25 mask pixels in the patch. Patch score = whole-patch Mahalanobis >= tau; pixel mask = overlap-averaged heatmap >= same tau, then morphology.
5. Zero denominators: patch/pixel precision, recall, F1 default to 0 when undefined (committed). So fabric IDs 04 and 06 zeros may be undefined-or-missed; also check 0011_006_02 (patch hits, empty pixel mask) and masks for zero-GT.
6. Averaging: mean per image over 13 rows (recomputed: overall and all 7 fabric rows match your table within rounding). Not pooled, not fabric-weighted. Pixel F1 = Dice only if the implementation uses the same masks (committed code: yes).
7. Orientation: AFID is 4096 wide x 256 high; committed config names `STRIP_HEIGHT=4096, STRIP_WIDTH=256` and the synthetic figure is drawn 4096 tall. Print `img.shape` on a real file.
8. **Figure provenance**: `Results/figures/inspection_Fabric01_Plain_broken_end.png` is synthetic (generator-style rectangle on synthetic texture; committed CSV row has 2400 GT px = 400x6, the generator's injection; shows tau = 3.45, which is not universal). Use it only as a labelled synthetic methodology demo (Slide 4). For Results use a real AFID figure or leave the placeholder.
9. Fabric-name mapping (ID <-> Plain/Twill/...) unverified; AFID page gives fabric codes 00-06 but no weave names. Report uses IDs.
10. Completed vs planned: saved per-image masks/overlays, confidence score, image-level metrics, adaptive-scale comparison, Gabor, 64x64 refinement are **not** done/confirmed. 5 Oct script question not supplied.
11. Earlier-run comparison (18 % -> 79.83 %, +61.8 pp) is between different configs/coverage: say "preliminary", not "improvement from adaptivity". Specificity 82.67 % (FPR 17.33 %) comes from the screenshot of an earlier run; confirm it belongs to the latest run before use.

Preliminary defect-code view (13 images; confounded with fabric; mean per-image %):

| Code | Official name | n | Patch F1 | Pixel recall | Pixel IoU |
|---|---|---|---|---|---|
| 002 | broken end | 4 | 49.09 | 75.00 | 0.82 |
| 006 | broken yarn | 2 | 18.18 | 50.00 | 0.28 |
| 010 | broken pick | 1 | 40.00 | 66.36 | 0.68 |
| 016 | weft curling | 1 | 0.00 | 0.00 | 0.00 |
| 019 | fuzzy ball | 4 | 0.00 | 0.00 | 0.00 |
| 027 | knot | 1 | 80.00 | 100.00 | 0.20 |

(019 samples are all fabric IDs 04 and 06, so "fuzzy ball" and those fabrics are inseparable here.) Full rows: `Results/sample_level_benchmark_13.csv`.

## E. Submission list
- GitHub repo `DIP_2026_17_15` (naming rule is `DIP_2026_ProjectNo_GroupNumber`; confirm project no. 17): dirs `Mid_Sem_Report/`, `End_Sem_Report/` (empty; add `.gitkeep`), `Results/`, `Codes/`. Current repo has `reports/` for weekly reports, which is fine as the weekly submission.
- `Mid_Sem_Report/`: `main.tex`, `references.bib`, figure files, final `main.pdf` (rename e.g. `DIP_2026_17_15_MidSem.pdf`); do not submit `TEAM_NOTES.md`.
- `Results/`: real-run metrics CSV, `sample_level_benchmark_13.csv`, real AFID figures; flag/remove synthetic outputs (`metrics_summary.csv`, `fabric_summary.csv`, inspection PNGs) or move them to `Results/synthetic_demo/`.
- `Codes/`: the upgraded code that produced the 13-image run + `requirements.txt`.
- Google Classroom (POC only): report PDF + PPT (6-8 slides, max 10; individual contributions slide).
- Weekly progress is a separate submission (Saturday 5 PM).
