"""Builds the LaTeX tables of the End-Semester report from Results/ (run from the repository root)."""

import json
from pathlib import Path

import pandas as pd

RESULTS = Path("Results")
OUT = Path("End_Sem_Report/tables")
END = "\\bottomrule\n"  # after \input a \bottomrule would be a misplaced \noalign


def pct(x) -> str:
    return "--" if x is None or pd.isna(x) else f"{100 * x:.1f}"


def num(x) -> str:
    return "--" if x is None or pd.isna(x) else f"{x:.3f}"


def tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("+", r"+\allowbreak{}")


def progression() -> str:
    """Strips, patch F1, pooled pixel F1 / IoU, mean pixel IoU, image AUROC, across the project."""
    legacy = pd.read_csv(RESULTS / "legacy" / "metrics_summary.csv")
    abl = pd.read_csv(RESULTS / "ablation" / "summary.csv").set_index("variant")
    oc = json.loads((RESULTS / "final" / "summary.json").read_text())
    sv = json.loads((RESULTS / "final_supervised" / "summary.json").read_text())
    rows = [
        ("Mid-sem (13 strips)", len(legacy), legacy["patch_f1"].mean(), None, None,
         legacy["pixel_iou"].mean(), None),
        ("Week 5 one-class", abl.loc["week 5 setting", "n_images"], abl.loc["week 5 setting", "patch_f1"],
         abl.loc["week 5 setting", "pooled_pixel_dice"], abl.loc["week 5 setting", "pooled_pixel_iou"],
         abl.loc["week 5 setting", "pixel_iou"], abl.loc["week 5 setting", "image_auroc"]),
        ("Week 7 one-class", oc["n_images"], oc["patch_f1"], oc["pooled_pixel_dice"], oc["pooled_pixel_iou"],
         oc["pixel_iou"], oc["image_auroc"]),
        ("Supervised (final)", sv["n_images"], sv["patch_f1"], sv["pooled_pixel_dice"], sv["pooled_pixel_iou"],
         sv["pixel_iou"], sv.get("image_auroc")),
    ]
    return "\n".join(f"{n} & {int(k)} & {pct(a)} & {pct(b)} & {pct(c)} & {pct(d)} & {num(e)} \\\\"
                     for n, k, a, b, c, d, e in rows) + "\n" + END


def levels() -> str:
    """Final supervised precision / recall / F1 / IoU at every level."""
    s = json.loads((RESULTS / "final_supervised" / "summary.json").read_text())

    def iou(p, r):
        return None if p is None or r is None or pd.isna(p) or pd.isna(r) or p + r == 0 else p * r / (p + r - p * r)

    m = pd.read_csv(RESULTS / "final_supervised" / "metrics_summary.csv")
    m = m[m["is_defective"].astype(bool) & (m["sample_id"] != "0080_036_01.png")]  # 48 % of all defect pixels
    tp, pr, gt = m["pixel_tp"].sum(), m["pred_defect_pixels"].sum(), m["gt_defect_pixels"].sum()
    rows = [
        ("Pixel, pooled", s["pooled_pixel_precision"], s["pooled_pixel_recall"], s["pooled_pixel_dice"],
         s["pooled_pixel_iou"]),
        (r"Pixel, pooled$^\dagger$", tp / pr, tp / gt, 2 * tp / (pr + gt), tp / (pr + gt - tp)),
        (r"Pixel, $\pm3$\,px", s["pooled_pixel_tol3_precision"], s["pooled_pixel_tol3_recall"],
         s["pooled_pixel_tol3_f1"], None),
        ("Pixel, mean/strip", s["pixel_precision"], s["pixel_recall"], s["pixel_dice"], s["pixel_iou"]),
        (r"$16^2$ window", s["pooled_window16_precision"], s["pooled_window16_recall"], s["pooled_window16_f1"],
         s["pooled_window16_iou"]),
        (r"$32^2$ window", s["pooled_window32_precision"], s["pooled_window32_recall"], s["pooled_window32_f1"],
         s["pooled_window32_iou"]),
        (r"$256^2$ patch", s["pooled_patch_precision"], s["pooled_patch_recall"], s["pooled_patch_f1"],
         iou(s["pooled_patch_precision"], s["pooled_patch_recall"])),
        ("Image (strip)", s["image_precision"], s["image_recall"], s["image_f1"], None),
    ]
    return "\n".join(f"{n} & {pct(p)} & {pct(r)} & {pct(f)} & {pct(i)} \\\\" for n, p, r, f, i in rows) + "\n" + END


def ablation() -> str:
    path = RESULTS / "ablation_supervised" / "summary.csv"
    one = pd.read_csv(RESULTS / "ablation" / "summary.csv").set_index("variant")
    if path.exists():
        rows = [(tex(r.variant), r.pooled_pixel_dice, r.pooled_pixel_iou, r.pixel_iou, r.image_auroc)
                for r in pd.read_csv(path).itertuples()]
    else:
        s = json.loads((RESULTS / "final_supervised" / "summary.json").read_text())
        rows = [("Supervised (final)", s["pooled_pixel_dice"], s["pooled_pixel_iou"], s["pixel_iou"], s["image_auroc"])]
    for v, name in (("default", "One-class, per-fabric"), ("reference: global (fixed)", "One-class, global"),
                    ("baseline: one-class svm", "One-class SVM"), ("baseline: isolation forest", "Isolation Forest")):
        r = one.loc[v]
        rows.append((name, r["pooled_pixel_dice"], r["pooled_pixel_iou"], r["pixel_iou"], r["image_auroc"]))
    return "\n".join(f"{n} & {pct(a)} & {pct(b)} & {pct(c)} & {num(d)} \\\\" for n, a, b, c, d in rows) + "\n" + END


def per_defect() -> str:
    m = pd.read_csv(RESULTS / "final_supervised" / "metrics_summary.csv")
    m = m[m["is_defective"]]
    rows = []
    for k, g in m.groupby("defect_type"):
        tp = g["pixel_tp"].sum()
        fp = g["pred_defect_pixels"].sum() - tp
        fn = g["gt_defect_pixels"].sum() - tp
        rows.append(f"{tex(k)} & {len(g)} & {pct(tp / (tp + fp) if tp + fp else None)} & "
                    f"{pct(tp / (tp + fn) if tp + fn else None)} & {pct(2 * tp / (2 * tp + fp + fn))} & "
                    f"{pct(tp / (tp + fp + fn))} & {pct(g['image_pred'].mean())} \\\\")
    return "\n".join(rows) + "\n" + END


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "progression.tex").write_text(progression())
    (OUT / "levels.tex").write_text(levels())
    (OUT / "ablation.tex").write_text(ablation())
    (OUT / "defect.tex").write_text(per_defect())
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
