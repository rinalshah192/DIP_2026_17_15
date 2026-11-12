"""Collects every results table into one Excel workbook: Results/AFID_results.xlsx.

Run from the repository root after the runs:  python Codes/scripts/export_excel.py
Needs openpyxl (pip install -e "Codes[excel]").
"""

import json
from pathlib import Path

import pandas as pd

RESULTS = Path("Results")


def summary_rows(folder: Path, name: str) -> pd.DataFrame:
    s = json.loads((folder / "summary.json").read_text())
    return pd.DataFrame([{"run": name, **s}])


def main() -> None:
    sheets = {}
    runs = {"supervised (final)": RESULTS / "final_supervised", "one-class (week 7)": RESULTS / "final"}
    sheets["summary"] = pd.concat([summary_rows(f, n) for n, f in runs.items() if (f / "summary.json").exists()])
    for name, folder in runs.items():
        tag = "sup" if name.startswith("supervised") else "oc"
        for csv, sheet in (("metrics_summary.csv", "per_image"), ("fabric_summary.csv", "per_fabric"),
                           ("defect_summary.csv", "per_defect"), ("image_level_by_fabric.csv", "image_by_fabric"),
                           ("thresholds.csv", "thresholds")):
            if (folder / csv).exists():
                sheets[f"{tag}_{sheet}"] = pd.read_csv(folder / csv)
    for ablation, tag in (("ablation_supervised", "sup_ablation"), ("ablation", "oc_ablation")):
        if (RESULTS / ablation / "summary.csv").exists():
            sheets[tag] = pd.read_csv(RESULTS / ablation / "summary.csv")
    out = RESULTS / "AFID_results.xlsx"
    with pd.ExcelWriter(out) as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    print(f"wrote {out} with sheets: {', '.join(sheets)}")


if __name__ == "__main__":
    main()
