import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ml_project" / "data"
CURATED = ROOT / "ml_project" / "curated_data"
REPORTS = ROOT / "ml_project" / "reports"


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


def test_sn_targets_expose_stress_metric_review_state():
    targets = read_csv(DATA / "sn_digitisation_targets.csv")
    required = {"stress_metric_type", "axis_scale_x", "axis_scale_y", "runout_encoding"}
    assert required.issubset(set(targets[0]))

    for row in targets:
        assert row["stress_metric_type"] in {"unknown", "stress_amplitude", "maximum_stress", "stress_range"}
        if row["stress_metric_type"] == "unknown":
            assert row["review_status"] == "needs_review"


def test_sn_preparation_audit_report_exists_and_records_gates():
    summary_path = REPORTS / "sn_digitisation_audit_summary.json"
    queue_path = REPORTS / "sn_pdf_review_queue.csv"
    assert summary_path.exists()
    assert queue_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["pdf_count"] == 36
    assert summary["registered_sn_targets"] >= 21
    assert summary["reviewed_sn_points"] >= 38
    assert "stress_metric_unknown" in summary["blocking_gates"]
    assert "figure_identity_unverified" in summary["blocking_gates"]

    queue = read_csv(queue_path)
    assert queue
    for row in queue:
        assert row["source_id"]
        assert row["pdf"]
        assert row["review_action"]


def test_sn_rendered_review_page_manifest_exists():
    manifest_path = REPORTS / "sn_rendered_review_pages.csv"
    assert manifest_path.exists()
    rows = read_csv(manifest_path)
    assert rows
    for row in rows:
        assert row["source_id"]
        assert row["pdf"]
        assert row["source_page"]
        assert row["local_review_image"]
