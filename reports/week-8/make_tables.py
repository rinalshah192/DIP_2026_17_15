"""Builds the LaTeX tables of the Week 8 report from Results/ (run from the repository root)."""

import json
from pathlib import Path

import pandas as pd

RESULTS = Path("Results")
OUT = Path("reports/week-8/tables")
FINAL = RESULTS / "final_supervised"
ONE_CLASS = RESULTS / "final"
BIG_STRIP = "0080_036_01.png"  # the weft crack whose mask covers 233k px


def pct(x) -> str:
    return "--" if x is None or pd.isna(x) else f"{100 * x:.2f}"


def num(x, d=3) -> str:
    return "--" if x is None or pd.isna(x) else f"{x:.{d}f}"


def tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("#", r"\#").replace("+", r"+\allowbreak{}")


def headline() -> str:
    """Precision / recall / F1 / IoU at every level: one-class Week-7 default vs final supervised."""
    oc = json.loads((ONE_CLASS / "summary.json").read_text())
    sv = json.loads((FINAL / "summary.json").read_text())

    def iou(p, r):  # IoU from precision and recall: TP / (TP + FP + FN)
        return None if p is None or r is None or pd.isna(p) or pd.isna(r) or p + r == 0 else p * r / (p + r - p * r)

    # One weft-crack strip (0080_036_01) holds 48 % of all AFID defect pixels; show pooled metrics without it too
    for s_, folder in ((oc, ONE_CLASS), (sv, FINAL)):
        m = pd.read_csv(folder / "metrics_summary.csv")
        m = m[m["is_defective"].astype(bool) & (m["sample_id"] != BIG_STRIP)]
        tp, pred, gt = m["pixel_tp"].sum(), m["pred_defect_pixels"].sum(), m["gt_defect_pixels"].sum()
        s_.update(x_precision=tp / pred if pred else None, x_recall=tp / gt if gt else None,
                  x_f1=2 * tp / (pred + gt) if pred + gt else None, x_iou=tp / (pred + gt - tp) if pred + gt else None)
    rows = [
        ("Pixel (pooled)", "pooled_pixel_precision", "pooled_pixel_recall", "pooled_pixel_dice", "pooled_pixel_iou"),
        (r"Pixel (pooled, w/o 0080\_036\_01)", "x_precision", "x_recall", "x_f1", "x_iou"),
        (r"Pixel (pooled, $\pm3$\,px)", "pooled_pixel_tol3_precision", "pooled_pixel_tol3_recall", "pooled_pixel_tol3_f1", None),
        ("Pixel (mean per strip)", "pixel_precision", "pixel_recall", "pixel_dice", "pixel_iou"),
        (r"$16\times16$ window (pooled)", "pooled_window16_precision", "pooled_window16_recall",
         "pooled_window16_f1", "pooled_window16_iou"),
        (r"$32\times32$ window (pooled)", "pooled_window32_precision", "pooled_window32_recall",
         "pooled_window32_f1", "pooled_window32_iou"),
        (r"$256\times256$ patch (pooled)", "pooled_patch_precision", "pooled_patch_recall", "pooled_patch_f1", None),
        ("Image (strip)", "image_precision", "image_recall", "image_f1", None),
    ]
    out = []
    for name, p, r, f, i in rows:
        cells = []
        for s in (oc, sv):
            pv, rv, fv = s.get(p), s.get(r), s.get(f)
            iv = s.get(i) if i else (None if name.startswith("Image") or "px)" in name else iou(pv, rv))
            cells += [pct(pv), pct(rv), pct(fv), pct(iv)]
        out.append(" & ".join([name] + cells) + r" \\")
    return "\n".join(out) + "\n"


def ablation() -> str:
    path = RESULTS / "ablation_supervised" / "summary.csv"
    if not path.exists():
        return r"\emph{ablation run in progress} & -- & -- & -- & -- & -- & -- & -- \\" + "\n"
    d = pd.read_csv(path)
    rows = []
    for r in d.itertuples():
        rows.append(" & ".join([tex(r.variant), pct(r.pooled_pixel_precision), pct(r.pooled_pixel_recall),
                                pct(r.pooled_pixel_dice), pct(r.pooled_pixel_iou), pct(r.pixel_iou),
                                num(getattr(r, "window16_auroc", float("nan"))), num(r.image_auroc)]) + r" \\")
    return "\n".join(rows) + "\n"


def breakdown(key: str) -> str:
    m = pd.read_csv(FINAL / "metrics_summary.csv")
    m = m[m["is_defective"]]
    rows = []
    for k, g in m.groupby(key):
        tp, fp = g["pixel_tp"].sum(), g["pred_defect_pixels"].sum() - g["pixel_tp"].sum()
        fn = g["gt_defect_pixels"].sum() - g["pixel_tp"].sum()
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
        iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        rows.append(" & ".join([tex(k), str(len(g)), pct(prec), pct(rec), pct(f1), pct(iou),
                                pct(g["pixel_iou"].mean()), num(g["window16_auroc"].mean()),
                                pct(g["image_pred"].mean())]) + r" \\")
    return "\n".join(rows) + "\n"


def thresholds() -> str:
    d = pd.read_csv(FINAL / "thresholds.csv")
    rows = []
    for r in d.itertuples():
        hough = "off" if pd.isna(r.hough) or r.hough in ("None", "") else str(r.hough)
        rows.append(" & ".join([str(r.fold), str(r.map), "pixel stage" if r.use_pixel else "window map",
                                num(r.gate, 2), num(r.pixel, 2), tex(hough), num(r.patch, 2), num(r.image, 2)]) + r" \\")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    end = "\\bottomrule\n"  # after \input a \bottomrule would be a misplaced \noalign
    (OUT / "headline.tex").write_text(headline() + end)
    (OUT / "ablation.tex").write_text(ablation() + end)
    (OUT / "fabric.tex").write_text(breakdown("fabric_type") + end)
    (OUT / "defect.tex").write_text(breakdown("defect_type") + end)
    (OUT / "thresholds.tex").write_text(thresholds() + end)
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
