import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ml_project" / "data"
CURATED = ROOT / "ml_project" / "curated_data"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sn_digitisation_targets_are_source_traceable():
    targets = read_csv(DATA / "sn_digitisation_targets.csv")
    sources = {row["source_id"] for row in read_csv(CURATED / "sources.csv")}
    source_files = {(row["source_id"], row["filename"]) for row in read_csv(CURATED / "source_files.csv")}

    assert targets
    for row in targets:
        assert row["target_id"]
        assert row["source_id"] in sources
        assert (row["source_id"], row["source_pdf"]) in source_files
        assert row["digitisation_status"]
        assert row["review_status"]


def test_sn_point_table_has_auditable_point_schema():
    with (DATA / "sn_curve_points.csv").open(newline="", encoding="utf-8") as f:
        fields = next(csv.reader(f))

    required = {
        "sn_point_id",
        "target_id",
        "source_id",
        "source_pdf",
        "source_page",
        "figure_id",
        "curve_id",
        "stress_amplitude_MPa",
        "max_stress_MPa",
        "cycles_to_failure",
        "stress_ratio_R",
        "test_temperature_C",
        "build_orientation",
        "surface_condition",
        "heat_treatment_class",
        "data_origin",
        "data_status",
        "review_status",
    }
    assert required.issubset(set(fields))
