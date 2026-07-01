#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_project.ht_advisor.ontology import classify_ht_text, normalize_ht_class
from ml_project.ht_advisor.paths import CURATED_DATA_DIR, EXTRACTED_DATA_DIR
from ml_project.ht_advisor.provenance import write_csv


def _clean(v: str | None) -> str:
    if v is None:
        return ""
    v = str(v).strip()
    return "" if v in {"-", "None", "nan"} else v


def main() -> None:
    CURATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed = EXTRACTED_DATA_DIR / "excel_seed_mechanical_dataset.csv"
    evidence = []
    conditions = []
    recipes = []
    steps = []
    measurements = []
    decisions = []

    with seed.open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f), start=1):
            if not any(_clean(row.get(k)) for k in ["UTS_MPa", "YS_MPa", "elongation_pct", "hardness_HV"]):
                continue
            source_id = f"EXCEL_REF_{_clean(row.get('ref_id')) or idx}"
            evidence_id = f"EVID_EXCEL_{idx:04d}"
            condition_id = f"COND_{idx:04d}"
            ht_id = f"HT_{idx:04d}"
            text = " ".join([_clean(row.get("alloy_am_orientation")), _clean(row.get("heat_treatment_text"))]).strip()
            ht_class = normalize_ht_class(_clean(row.get("ht_class"))) or classify_ht_text(text)

            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "source_id": source_id,
                    "file_id": "FILE_EXCEL_SEED",
                    "page": "",
                    "table_or_figure_id": "ML LPBF 718 Sheet1",
                    "row_or_curve_label": row["source_row"],
                    "excerpt": text,
                    "raw_value_text": "; ".join(f"{k}={row.get(k, '')}" for k in ["UTS_MPa", "YS_MPa", "elongation_pct", "hardness_HV"]),
                    "extraction_method": "excel_seed_curated",
                    "candidate_file": "excel_seed_mechanical_dataset.csv",
                    "candidate_row_id": str(idx),
                    "reviewer_status": "curated_reviewed",
                }
            )
            conditions.append(
                {
                    "condition_id": condition_id,
                    "source_id": source_id,
                    "alloy": "Inconel 718",
                    "am_process": "LPBF/SLM",
                    "machine_make_model": "",
                    "eos_like_flag": "",
                    "powder_supplier": "",
                    "composition_source": "",
                    "build_orientation": "vertical" if "vertical" in text.lower() else "",
                    "process_parameters": "",
                    "surface_state": "",
                    "specimen_geometry": "",
                    "evidence_id": evidence_id,
                }
            )
            recipes.append(
                {
                    "ht_id": ht_id,
                    "condition_id": condition_id,
                    "ht_class": ht_class,
                    "recipe_label": text,
                    "include_in_model": "true",
                    "curation_status": "curated_reviewed",
                    "evidence_id": evidence_id,
                }
            )
            steps.append(
                {
                    "ht_step_id": f"HTSTEP_{idx:04d}_001",
                    "ht_id": ht_id,
                    "step_order": "1",
                    "step_type": "recipe_text",
                    "temperature_C": "",
                    "time_h": "",
                    "pressure_MPa": "",
                    "atmosphere": "",
                    "cooling_or_quench": "",
                    "thermal_dose_feature": "",
                    "evidence_id": evidence_id,
                }
            )
            for prop, unit in [("UTS_MPa", "MPa"), ("YS_MPa", "MPa"), ("elongation_pct", "%"), ("hardness_HV", "HV")]:
                val = _clean(row.get(prop))
                if not val:
                    continue
                measurements.append(
                    {
                        "measurement_id": f"MEAS_{idx:04d}_{prop}",
                        "condition_id": condition_id,
                        "ht_id": ht_id,
                        "property_type": prop,
                        "value": val,
                        "unit": unit,
                        "test_temperature_C": "",
                        "test_standard": "",
                        "n": "",
                        "statistic_type": "reported",
                        "uncertainty": "",
                        "data_origin": "excel_seed",
                        "evidence_id": evidence_id,
                        "curation_status": "curated_reviewed",
                    }
                )
            decisions.append(
                {
                    "decision_id": f"DEC_{idx:04d}",
                    "object_type": "excel_seed_row",
                    "object_id": row["source_row"],
                    "decision": "include",
                    "reason": "Seed row has at least one reported mechanical property and AM/LPBF context from workbook.",
                    "reviewer": "HT-Advisor seed builder",
                    "date": "",
                    "confidence": "medium",
                    "notes": "Requires source-level PDF verification before final publication claims.",
                }
            )

    write_csv(CURATED_DATA_DIR / "evidence_spans.csv", evidence)
    write_csv(CURATED_DATA_DIR / "manufacturing_conditions.csv", conditions)
    write_csv(CURATED_DATA_DIR / "heat_treatment_recipes.csv", recipes)
    write_csv(CURATED_DATA_DIR / "heat_treatment_steps.csv", steps)
    write_csv(CURATED_DATA_DIR / "mechanical_measurements.csv", measurements)
    write_csv(CURATED_DATA_DIR / "curation_decisions.csv", decisions)
    print(f"Wrote curated seed tables to {CURATED_DATA_DIR}")


if __name__ == "__main__":
    main()
