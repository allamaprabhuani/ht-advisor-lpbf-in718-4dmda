import pandas as pd

from ml_project.ht_advisor.sn_fatigue import (
    build_reviewed_sn_points,
    fit_basquin_models,
    make_prediction_grid,
)


def test_build_reviewed_sn_points_creates_traceable_marker_dataset():
    points = build_reviewed_sn_points()

    assert not points.empty
    assert {
        "sn_point_id",
        "source_id",
        "source_pdf",
        "source_page",
        "figure_id",
        "cycles_to_failure",
        "stress_amplitude_MPa",
        "stress_ratio_R",
        "heat_treatment_class",
        "data_status",
        "review_status",
    }.issubset(points.columns)
    assert len(points) >= 30
    assert points["review_status"].eq("reviewed").all()
    assert points["data_status"].str.contains("marker_digitised").any()
    assert points["data_origin"].str.contains("sn_priority_rendered").any()
    assert points["stress_metric_type"].eq("stress_amplitude").all()
    assert points["cycles_to_failure"].astype(float).gt(0).all()
    assert points["stress_amplitude_MPa"].astype(float).between(150, 900).all()


def test_fit_basquin_models_keeps_conditions_separate_and_excludes_runouts():
    points = build_reviewed_sn_points()
    summary = fit_basquin_models(points, minimum_failures=3)

    assert not summary.empty
    assert {"condition_id", "source_id", "heat_treatment_class", "stress_ratio_R", "n_failures", "n_runouts", "basquin_slope_b"}.issubset(summary.columns)
    assert summary["n_failures"].ge(3).all()
    assert summary["basquin_slope_b"].lt(0).all()
    assert summary["model_status"].eq("trained_condition_specific_basquin").all()
    assert (summary["n_runouts"] > 0).any()
    assert summary.groupby("stress_ratio_R")["condition_id"].nunique().ge(1).all()


def test_prediction_grid_is_monotonic_and_traceable_to_model_summary():
    points = build_reviewed_sn_points()
    summary = fit_basquin_models(points, minimum_failures=3)
    grid = make_prediction_grid(summary, points_per_condition=12)

    assert not grid.empty
    assert {"condition_id", "cycles_to_failure", "stress_amplitude_MPa", "stress_lower_MPa", "stress_upper_MPa"}.issubset(grid.columns)
    for _, subset in grid.groupby("condition_id"):
        ordered = subset.sort_values("cycles_to_failure")
        assert ordered["stress_amplitude_MPa"].is_monotonic_decreasing
        assert ordered["stress_lower_MPa"].le(ordered["stress_amplitude_MPa"]).all()
        assert ordered["stress_upper_MPa"].ge(ordered["stress_amplitude_MPa"]).all()


def test_fit_basquin_models_rejects_underpopulated_condition():
    points = pd.DataFrame(
        [
            {
                "source_id": "SRC",
                "condition_id": "SRC_DA_R0p1",
                "heat_treatment_class": "DA",
                "stress_ratio_R": "0.1",
                "runout_flag": False,
                "cycles_to_failure": 10000,
                "stress_amplitude_MPa": 600,
            },
            {
                "source_id": "SRC",
                "condition_id": "SRC_DA_R0p1",
                "heat_treatment_class": "DA",
                "stress_ratio_R": "0.1",
                "runout_flag": False,
                "cycles_to_failure": 100000,
                "stress_amplitude_MPa": 450,
            },
        ]
    )

    summary = fit_basquin_models(points, minimum_failures=3)

    assert summary.empty
