import pandas as pd

import ml_project.ht_advisor.dashboard_data as dashboard_data
from ml_project.ht_advisor.dashboard_data import build_property_tradeoff_rows, parse_process_window
from ml_project.ht_advisor.dashboard_data import (
    build_recommendation_contribution_rows,
    build_route_radar_rows,
    build_thermal_cycle_rows,
)


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


def test_build_thermal_cycle_rows_creates_ordered_time_temperature_profile():
    rows = build_thermal_cycle_rows(
        "ST_DA",
        "Solution treatment about 980-1095 C for 1-2 h, then ageing about 720 C/8 h and 620 C/8 h",
    )

    assert not rows.empty
    assert {"elapsed_h", "temperature_C", "stage", "ht_class"}.issubset(rows.columns)
    assert rows["elapsed_h"].is_monotonic_increasing
    assert rows["temperature_C"].max() == 1095
    assert rows["temperature_C"].iloc[0] == 25
    assert rows["temperature_C"].iloc[-1] == 25


def test_build_thermal_cycle_segment_rows_labels_process_steps_for_colored_plotting():
    rows = dashboard_data.build_thermal_cycle_segment_rows("ST_DA", "980 C for 1 h; 720 C for 8 h; 620 C for 8 h")

    assert not rows.empty
    assert {"segment_id", "segment_label", "segment_type", "elapsed_h", "temperature_C", "ht_class"}.issubset(rows.columns)
    assert rows.groupby("segment_id").size().eq(2).all()
    assert rows["segment_label"].drop_duplicates().tolist() == [
        "Ramp to solution treatment",
        "Solution treatment hold",
        "Transition to first ageing",
        "First ageing hold",
        "Transition to second ageing",
        "Second ageing hold",
        "Final cooling",
    ]


def test_build_route_radar_rows_normalises_property_and_evidence_axes():
    rows = pd.DataFrame(
        [
            {
                "ht_class": "ST_DA",
                "ml_assisted_score": 0.75,
                "evidence_count_seed": 6,
                "predicted_UTS_MPa": 1450,
                "predicted_YS_MPa": 1200,
                "predicted_elongation_pct": 10,
            },
            {
                "ht_class": "DA",
                "ml_assisted_score": 0.55,
                "evidence_count_seed": 2,
                "predicted_UTS_MPa": 1380,
                "predicted_YS_MPa": 1040,
                "predicted_elongation_pct": 18,
            },
        ]
    )

    radar = build_route_radar_rows(rows)

    assert {"ht_class", "axis", "value"}.issubset(radar.columns)
    assert set(radar["axis"]) == {"UTS", "Yield strength", "Elongation", "Evidence support", "Recommendation index"}
    assert radar["value"].between(0, 100).all()


def test_build_recommendation_contribution_rows_splits_evidence_and_property_terms():
    row = {
        "adjusted_score": 0.7,
        "ml_property_index": 0.8,
        "ml_assisted_score": 0.73,
    }

    rows = build_recommendation_contribution_rows(row)

    assert list(rows["term"]) == ["Evidence and feasibility", "Calibrated property contribution", "Recommendation index"]
    assert rows.loc[0, "measure"] == "relative"
    assert rows.loc[2, "measure"] == "total"
    assert round(rows.loc[2, "value"], 2) == 0.73
