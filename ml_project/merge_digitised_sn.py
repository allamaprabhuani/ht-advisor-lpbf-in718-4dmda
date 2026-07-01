#!/usr/bin/env python3
"""
Merge digitised S-N point CSV files into ml_project/data/sn_curve_points.csv.

This script intentionally does not digitise plots. It standardises outputs from
digitize_sn.py once each figure has been calibrated and reviewed.
"""
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TARGETS = DATA / "sn_digitisation_targets.csv"
OUT = DATA / "sn_curve_points.csv"


def load_targets():
    with TARGETS.open(newline="", encoding="utf-8") as f:
        return {row["target_id"]: row for row in csv.DictReader(f)}


def output_fields():
    with OUT.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def as_float(value):
    try:
        if value == "":
            return ""
        return float(value)
    except Exception:
        return ""


def log10_or_blank(value):
    value = as_float(value)
    if value == "" or value <= 0:
        return ""
    return math.log10(value)


def find_digitised_files():
    for path in sorted((ROOT / "figures").glob("**/*.csv")):
        if path.name == "SN_candidates.csv":
            continue
        yield path


def infer_target_id(path):
    # Preferred convention: target id appears in parent or file stem.
    name = f"{path.parent.name}/{path.stem}"
    for target_id in load_targets():
        if target_id in name:
            return target_id
    return ""


def main():
    targets = load_targets()
    fields = output_fields()
    rows = []
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as f:
            for existing in csv.DictReader(f):
                origin = existing.get("source_type", "")
                if origin and origin != "literature_digitised":
                    rows.append(existing)

    for csv_path in find_digitised_files():
        target_id = infer_target_id(csv_path)
        if not target_id:
            print(f"skip, cannot infer target_id: {csv_path}")
            continue
        target = targets[target_id]
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, point in enumerate(reader, start=1):
                cycles = point.get("x", "")
                stress = point.get("y", "")
                row = {field: "" for field in fields}
                row.update(
                    {
                        "sn_point_id": f"{target_id}-{idx:04d}",
                        "source_type": "literature_digitised",
                        "source_id": target["source_id"],
                        "source_pdf": target["source_pdf"],
                        "source_page": target["source_page"],
                        "figure_id": target["figure_id"],
                        "curve_id": "",
                        "condition_id": "",
                        "point_index": idx,
                        "alloy": target["material_alloy"],
                        "test_type": "fatigue",
                        "stress_metric_digitised": target["y_axis_label"],
                        "cycles_to_failure": cycles,
                        "log10_cycles_to_failure": log10_or_blank(cycles),
                        "data_origin": str(csv_path),
                        "data_status": "digitised_needs_metadata",
                        "notes": "Merged from digitize_sn.py output; condition metadata requires manual assignment.",
                    }
                )
                y_label = target["y_axis_label"].lower()
                if "amplitude" in y_label:
                    row["stress_amplitude_MPa"] = stress
                elif "maximum" in y_label or "max" in y_label:
                    row["max_stress_MPa"] = stress
                rows.append(row)

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
