import csv
import subprocess

from ml_project.ht_advisor.paths import CURATED_DATA_DIR, PROJECT_ROOT


def test_build_curated_seed_outputs_required_tables():
    subprocess.run(["python3", "ml_project/scripts/build_curated_seed.py"], cwd=PROJECT_ROOT, check=True)
    for name in ["evidence_spans.csv", "manufacturing_conditions.csv", "heat_treatment_recipes.csv", "heat_treatment_steps.csv", "mechanical_measurements.csv", "curation_decisions.csv"]:
        assert (CURATED_DATA_DIR / name).exists()


def test_curated_mechanical_rows_have_evidence():
    p = CURATED_DATA_DIR / "mechanical_measurements.csv"
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert all(r["evidence_id"] for r in rows)
    assert all(r["curation_status"] == "curated_reviewed" for r in rows)

