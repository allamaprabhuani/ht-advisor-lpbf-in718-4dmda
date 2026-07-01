import pandas as pd

from ml_project.ht_advisor.dashboard_data import build_property_tradeoff_rows, parse_process_window


def test_parse_process_window_extracts_temperature_range():
    row = parse_process_window("ST_DA", "Solution treatment about 980-1095 C for 1-2 h, then ageing about 720 C/8 h and 620 C/8 h")
    assert row["min_temperature_C"] == 620
    assert row["max_temperature_C"] == 1095
    assert row["process_family"] == "non-HIP"


def test_build_property_tradeoff_rows_returns_numeric_axes():
    recs = pd.DataFrame(
        [
            {"ht_class": "ST_DA", "target": "balanced", "allow_hip": False, "confidence_mode": "balanced", "rank": 1, "score": 0.75, "evidence_count_seed": 6},
            {"ht_class": "DA", "target": "balanced", "allow_hip": False, "confidence_mode": "balanced", "rank": 2, "score": 0.55, "evidence_count_seed": 2},
        ]
    )
    rows = build_property_tradeoff_rows(recs, "balanced", False, "balanced")
    assert list(rows["ht_class"]) == ["ST_DA", "DA"]
    assert rows["recommendation_index"].dtype.kind in "fi"
    assert rows["evidence_count"].tolist() == [6, 2]
