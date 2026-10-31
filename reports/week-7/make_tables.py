"""Builds the LaTeX tables of the Week 7 report from Results/ (run from the repository root)."""

from pathlib import Path

import pandas as pd

RESULTS = Path("Results")
OUT = Path("reports/week-7/tables")


def pct(x) -> str:
    return "--" if pd.isna(x) else f"{100 * x:.2f}"


def num(x, d=3) -> str:
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("#", r"\#").replace("+", r"+\allowbreak{}")


def ablation() -> str:
    d = pd.read_csv(RESULTS / "ablation" / "summary.csv")
    rows = []
    for r in d.itertuples():
        name = tex(r.variant)
        cells = [pct(r.patch_f1), pct(r.pooled_patch_f1), num(r.patch_auroc), pct(r.pixel_iou),
                 pct(r.pixel_recall), pct(r.image_accuracy), pct(r.image_recall), pct(r.image_specificity),
                 num(r.image_auroc)]
        line = " & ".join([name] + cells) + r" \\"
        rows.append(r"\textbf{" + name + "}" + line[len(name):] if r.variant == "default" else line)
    return "\n".join(rows) + "\n"


def by(key: str, file: str) -> str:
    d = pd.read_csv(RESULTS / "final" / file)
    m = pd.read_csv(RESULTS / "final" / "metrics_summary.csv")
    m = m[m["is_defective"]]
    auroc = m.groupby(key)["patch_auroc"].mean()
    det = m.groupby(key)["image_pred"].mean()
    rows = []
    for r in d.to_dict("records"):
        k = r[key]
        rows.append(" & ".join([tex(k), str(r["n_images"]), pct(r["patch_precision"]), pct(r["patch_recall"]),
                                pct(r["patch_f1"]), num(auroc.get(k)), pct(r["pixel_iou"]),
                                pct(r["pixel_recall"]), pct(det.get(k))]) + r" \\")
    return "\n".join(rows) + "\n"


def image_by_fabric() -> str:
    d = pd.read_csv(RESULTS / "final" / "image_level_by_fabric.csv")
    return "\n".join(" & ".join([tex(r.fabric_type), str(r.n_pos), str(r.n_neg), pct(r.accuracy), pct(r.recall),
                                 pct(r.specificity), num(r.auroc)]) + r" \\" for r in d.itertuples()) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    # \bottomrule lives here: after \input it would be a misplaced \noalign
    end = "\\bottomrule\n"
    (OUT / "ablation.tex").write_text(ablation() + end)
    (OUT / "fabric.tex").write_text(by("fabric_type", "fabric_summary.csv") + end)
    (OUT / "defect.tex").write_text(by("defect_type", "defect_summary.csv") + end)
    (OUT / "image_fabric.tex").write_text(image_by_fabric() + end)
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
