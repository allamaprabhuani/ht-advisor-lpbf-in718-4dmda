import pandas as pd
import subprocess
import sys

from ml_project.ht_advisor.sn_fatigue import (
    build_reviewed_sn_points,
    build_stress_ratio_screening_table,
    fit_basquin_models,
    goodman_equivalent_reversed_amplitude,
    make_prediction_grid,
    stress_amplitude_for_goodman_target,
)
from ml_project.scripts import train_sn_fatigue_model


def test_dashboard_sn_screening_import_does_not_require_opencv():
    script = """
import importlib.abc
import sys

class BlockCv2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("No module named 'cv2'")
        return None

sys.meta_path.insert(0, BlockCv2())
from ml_project.ht_advisor.sn_fatigue import build_stress_ratio_screening_table
print(build_stress_ratio_screening_table.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=".",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "build_stress_ratio_screening_table" in result.stdout


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
    assert {
        "condition_id",
        "source_id",
        "heat_treatment_class",
        "stress_ratio_R",
        "n_failures",
        "n_runouts",
        "basquin_slope_b",
        "censoring_method",
        "n_censored_constraints_satisfied",
        "runout_margin_log10_stress",
    }.issubset(summary.columns)
    assert summary["n_failures"].ge(3).all()
    assert summary["basquin_slope_b"].lt(0).all()
    assert summary["model_status"].eq("trained_condition_specific_censored_basquin_screening").all()
    assert summary["censoring_method"].str.contains("right-censored").all()
    assert (summary["n_runouts"] > 0).any()
    assert summary.groupby("stress_ratio_R")["condition_id"].nunique().ge(1).all()
    runout_models = summary[summary["n_runouts"].gt(0)]
    assert runout_models["n_censored_constraints_satisfied"].ge(1).all()


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


def test_sn_training_artifact_reports_stress_ratio_boundary(tmp_path):
    report = train_sn_fatigue_model.train_and_write_sn_artifacts(tmp_path)

    assert report["reviewed_points"] >= 30
    assert report["trained_condition_models"] >= 4
    assert "-1" in report["stress_ratio_groups"]
    assert report["local_r_ratio_predictor"] is False
    assert "No R = 0.1 fatigue-life predictor" in report["application_boundary"]
    assert report["runout_handling"].startswith("Runout markers are treated as right-censored")
    assert "censored" in report["model_family"].lower()


def test_goodman_stress_ratio_translation_keeps_r_minus_one_unchanged_and_reduces_positive_r_amplitude():
    assert goodman_equivalent_reversed_amplitude(700, stress_ratio_R=-1, uts_MPa=1350) == 700

    r01_equivalent = goodman_equivalent_reversed_amplitude(430, stress_ratio_R=0.1, uts_MPa=1350)
    assert 690 < r01_equivalent < 710

    translated = stress_amplitude_for_goodman_target(700, stress_ratio_R=0.1, uts_MPa=1350)
    assert 425 < translated < 435


def test_stress_ratio_screening_table_provides_multi_r_target_stresses_without_retraining_claim():
    points = build_reviewed_sn_points()
    summary = fit_basquin_models(points, minimum_failures=3)
    table = build_stress_ratio_screening_table(
        summary,
        stress_ratios=[-1, 0.1],
        target_cycles=[1_000_000],
        uts_MPa=1350,
    )

    assert not table.empty
    assert {
        "condition_id",
        "target_cycles",
        "screening_stress_ratio_R",
        "screening_stress_amplitude_MPa",
        "sigma_max_MPa",
        "sigma_min_MPa",
        "goodman_equivalent_R_minus_1_MPa",
        "screening_boundary",
    }.issubset(table.columns)
    assert table["source_curve_stress_ratio_R"].eq("-1").all()
    assert table["screening_boundary"].str.contains("Goodman").all()

    ht1 = table[table["condition_id"].eq("NEW10_HT1_Rminus1_RT")]
    rminus1 = float(ht1[ht1["screening_stress_ratio_R"].eq(-1.0)]["screening_stress_amplitude_MPa"].iloc[0])
    r01 = float(ht1[ht1["screening_stress_ratio_R"].eq(0.1)]["screening_stress_amplitude_MPa"].iloc[0])
    assert r01 < rminus1
