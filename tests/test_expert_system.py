import pandas as pd

from ml_project.ht_advisor.expert_system import (
    ManualInputContext,
    apply_manual_inputs,
    build_example_input_combinations,
    build_fatigue_validation_schedule,
    build_must_have_experiments,
    build_model_specification,
    build_printable_recommendation_report,
    build_raw_training_data_table,
    build_sn_training_status,
    generate_text_recommendation,
    select_recommendation_subset,
)


def test_manual_inputs_mark_routes_outside_local_furnace_limit():
    rows = pd.DataFrame(
        [
            {
                "ht_class": "ST_DA",
                "score": 0.78,
                "temperature_time_window": "Solution treatment about 980-1095 C for 1-2 h, then ageing about 720 C/8 h and 620 C/8 h",
                "recommendation_reason": "Non-HIP route.",
                "confidence": "medium",
                "evidence_count_seed": 6,
            },
            {
                "ht_class": "DA",
                "score": 0.65,
                "temperature_time_window": "Direct ageing around 720 C for 8 h plus 620 C for 8 h where applicable",
                "recommendation_reason": "Direct ageing route.",
                "confidence": "low",
                "evidence_count_seed": 2,
            },
        ]
    )

    adjusted = apply_manual_inputs(rows, ManualInputContext(furnace_limit_C=980, maximum_cycle_hours=20))

    st_da = adjusted.loc[adjusted["ht_class"] == "ST_DA"].iloc[0]
    da = adjusted.loc[adjusted["ht_class"] == "DA"].iloc[0]
    assert st_da["local_feasibility"] == "limited by selected furnace range"
    assert "exceeds the selected furnace range" in st_da["constraint_notes"]
    assert st_da["estimated_furnace_occupancy_h"] > st_da["estimated_cycle_hours"]
    assert "solution-treatment temperature exceeds" in st_da["metallurgical_rule_flags"]
    assert da["local_feasibility"] == "feasible under selected constraints"
    assert da["adjusted_rank"] == 1
    assert "No explicit solution treatment" in da["metallurgical_rule_flags"]


def test_manual_inputs_rank_by_selected_recipe_not_upper_literature_window():
    rows = pd.DataFrame(
        [
            {
                "ht_class": "ST_DA",
                "score": 0.78,
                "temperature_time_window": "Solution treatment about 980-1095 C for 1-2 h, then ageing about 720 C/8 h and 620 C/8 h",
                "selected_recipe_summary": "980 C for 1 h; 720 C for 8 h; 620 C for 8 h",
                "recommended_peak_temperature_C": 980,
                "recommended_total_hold_h": 17.0,
                "confidence": "medium",
                "evidence_count_seed": 12,
            },
            {
                "ht_class": "HA_ST_DA",
                "score": 0.76,
                "temperature_time_window": "Homogenisation around 1065-1100 C, solution treatment, then double ageing",
                "selected_recipe_summary": "1065 C for 1 h; 980 C for 1 h; 720 C for 8 h; 620 C for 8 h",
                "recommended_peak_temperature_C": 1065,
                "recommended_total_hold_h": 18.0,
                "confidence": "medium",
                "evidence_count_seed": 5,
            },
        ]
    )

    adjusted = apply_manual_inputs(rows, ManualInputContext(furnace_limit_C=980, maximum_cycle_hours=20))

    st_da = adjusted.loc[adjusted["ht_class"] == "ST_DA"].iloc[0]
    ha = adjusted.loc[adjusted["ht_class"] == "HA_ST_DA"].iloc[0]
    assert st_da["local_feasibility"] == "feasible under selected constraints"
    assert st_da["maximum_temperature_C"] == 980
    assert st_da["estimated_cycle_hours"] == 17.0
    assert ha["local_feasibility"] == "limited by selected furnace range"
    assert st_da["adjusted_rank"] == 1


def test_text_recommendation_reports_route_effects_and_stochastic_uncertainty():
    row = {
        "ht_class": "ST_DA",
        "temperature_time_window": "Solution treatment about 980-1095 C for 1-2 h, then ageing about 720 C/8 h and 620 C/8 h",
        "selected_recipe_summary": "980 C for 1 h; 720 C for 8 h; 620 C for 8 h",
        "local_feasibility": "feasible under selected constraints",
        "constraint_notes": "No constraint penalties were applied.",
        "recommendation_reason": "Literature-supported non-HIP route.",
        "confidence": "medium",
        "adjusted_score": 0.79,
    }
    text = generate_text_recommendation(row, ManualInputContext(target_life_cycles=1000000, stress_ratio_R=0.1, surface_condition="machined"))

    assert "ST_DA" in text
    assert "980 C for 1 h; 720 C for 8 h; 620 C for 8 h" in text
    assert "recommended validation recipe" in text
    assert "R = 0.1" in text
    assert "1,000,000 cycles" in text
    assert "precipitation strengthening" in text
    assert "stochastic" in text.lower()
    assert "local validation" in text.lower()


def test_model_specification_describes_inputs_outputs_and_uncertainty():
    spec = build_model_specification()

    assert "model_family" in spec
    assert "inputs" in spec
    assert "outputs" in spec
    assert "uncertainty_treatment" in spec
    assert "empirically calibrated parametric model" in spec["model_family"]
    assert "heat-treatment route" in " ".join(spec["outputs"])
    assert "estimated furnace occupancy" in " ".join(spec["outputs"])
    assert "delta-phase fraction" in " ".join(spec["uncertainty_treatment"])


def test_raw_training_data_table_keeps_reference_links_and_hashes():
    sources = pd.DataFrame(
        [
            {
                "source_id": "OA04",
                "title": "Heat treatment of LPBF Inconel 718",
                "doi": "",
                "url": "",
                "am_scope": "AM primary",
                "recommended_model_use": "yes",
            }
        ]
    )
    source_files = pd.DataFrame(
        [
            {
                "source_id": "OA04",
                "filename": "OA04.pdf",
                "sha256": "abc123",
                "local_path": "ml_project/corpus_pdfs/OA04.pdf",
                "download_status": "available",
            }
        ]
    )
    online_manifest = pd.DataFrame(
        [
            {
                "source_id": "OA04",
                "title": "Heat treatment of LPBF Inconel 718",
                "url": "https://example.org/paper.pdf",
                "source_type": "article_pdf",
                "focus": "heat-treatment",
            }
        ]
    )

    table = build_raw_training_data_table(sources, source_files, online_manifest)

    assert list(table.columns) == [
        "source_id",
        "title",
        "doi",
        "url",
        "am_scope",
        "recommended_model_use",
        "filename",
        "sha256",
        "local_path",
        "download_status",
    ]
    assert table.loc[0, "url"] == "https://example.org/paper.pdf"
    assert table.loc[0, "sha256"] == "abc123"


def test_must_have_experiments_include_baselines_microstructure_and_fatigue_caution():
    experiments = build_must_have_experiments("ST_DA", allow_hip=False)
    joined = " ".join(item["experiment"] + " " + item["reason"] for item in experiments)

    assert "as-built baseline" in joined
    assert "AMS-style" in joined
    assert "framework-recommended" in joined
    assert "SEM/EDS" in joined
    assert "fatigue" in joined.lower()
    assert any(item["priority"] == "required" for item in experiments)


def test_fatigue_validation_schedule_computes_r_ratio_stress_levels_without_predicting_life():
    schedule = build_fatigue_validation_schedule(stress_ratio_R=0.1, target_life_cycles=1000000)

    assert list(schedule.columns) == [
        "stress_amplitude_MPa",
        "stress_ratio_R",
        "sigma_max_MPa",
        "sigma_min_MPa",
        "sigma_mean_MPa",
        "goodman_equivalent_R_minus_1_MPa",
        "mean_stress_correction",
        "target_runout_cycles",
        "interpretation",
    ]
    assert schedule["stress_amplitude_MPa"].tolist() == [300, 350, 400, 450]
    first = schedule.iloc[0]
    assert first["sigma_max_MPa"] == 667
    assert first["sigma_min_MPa"] == 67
    assert first["sigma_mean_MPa"] == 367
    assert first["goodman_equivalent_R_minus_1_MPa"] > first["stress_amplitude_MPa"]
    assert "Goodman" in first["mean_stress_correction"]
    assert first["target_runout_cycles"] == 1000000
    assert schedule["interpretation"].str.contains("validation stress level, not predicted life").all()


def test_sn_training_status_disables_fatigue_life_prediction_without_reviewed_points():
    status = build_sn_training_status(
        sn_points=pd.DataFrame(columns=["source_id", "review_status"]),
        sn_targets=pd.DataFrame([{"source_id": "SRC001"}, {"source_id": "SRC002"}]),
    )

    assert status["sn_model_trained"] is False
    assert status["reviewed_point_rows"] == 0
    assert status["registered_targets"] == 2
    assert "S-N curves have not yet been trained" in status["status_message"]
    assert "Fatigue life is not predicted" in status["report_note"]


def test_sn_training_status_keeps_literature_fit_separate_from_local_life_prediction():
    status = build_sn_training_status(
        sn_points=pd.DataFrame(
            [{"source_id": "SRC", "review_status": "reviewed", "stress_ratio_R": "-1"} for _ in range(12)]
        ),
        sn_targets=pd.DataFrame([{"source_id": "SRC001"}]),
    )

    assert status["sn_model_trained"] is True
    assert "literature S-N screening module" in status["status_message"]
    assert "No local R = 0.1 fatigue-life predictor" in status["report_note"]
    assert "fitted fatigue-life model" not in status["status_message"]


def test_printable_report_includes_inputs_recipe_static_estimates_and_validation_boundary():
    schedule = build_fatigue_validation_schedule(stress_ratio_R=0.1, target_life_cycles=1000000)
    report = build_printable_recommendation_report(
        input_conditions={
            "Primary design objective": "fatigue",
            "Build orientation": "vertical",
            "Surface condition": "machined",
            "Maximum furnace temperature": "980 C",
        },
        top_row={
            "ht_class": "ST_DA",
            "selected_recipe_summary": "980 C for 1 h; 720 C for 8 h; 620 C for 8 h",
            "recommended_peak_temperature_C": 980,
            "recommended_total_hold_h": 17.0,
            "ml_assisted_score": 0.81,
            "confidence": "medium",
            "local_feasibility": "feasible under selected constraints",
            "predicted_UTS_MPa": 1393.6,
            "predicted_UTS_MPa_lower": 1371.4,
            "predicted_UTS_MPa_upper": 1415.9,
            "predicted_YS_MPa": 1089.4,
            "predicted_YS_MPa_lower": 1061.8,
            "predicted_YS_MPa_upper": 1116.9,
            "predicted_elongation_pct": 12.7,
            "predicted_elongation_pct_lower": 11.4,
            "predicted_elongation_pct_upper": 14.0,
        },
        context=ManualInputContext(target_life_cycles=1000000, stress_ratio_R=0.1),
        fatigue_schedule=schedule,
        sn_status={
            "sn_model_trained": False,
            "reviewed_point_rows": 0,
            "registered_targets": 21,
            "status_message": "S-N curves have not yet been trained.",
            "report_note": "Fatigue life is not predicted in the current release.",
        },
        experiments=build_must_have_experiments("ST_DA", allow_hip=False),
    )

    assert "Printable recommendation report" in report
    assert "Full input conditions" in report
    assert "ST_DA" in report
    assert "980 C for 1 h; 720 C for 8 h; 620 C for 8 h" in report
    assert "Expected static-property estimates" in report
    assert "UTS: 1393.6 MPa" in report
    assert "Fatigue validation schedule" in report
    assert "sigma_max = 667 MPa" in report
    assert "S-N training status" in report
    assert "S-N curves have not yet been trained" in report
    assert "Fatigue life is not predicted" in report
    assert "Must-have experimental validation" in report


def test_printable_report_contains_technician_heat_treatment_instruction_sheet():
    schedule = build_fatigue_validation_schedule(stress_ratio_R=0.1, target_life_cycles=1000000, stress_amplitudes_MPa=[300])
    report = build_printable_recommendation_report(
        input_conditions={
            "Initial material state": "EOS-like LPBF, machined",
            "Build orientation": "vertical",
            "Surface condition": "machined",
            "Representative section size": "thin section",
            "Maximum furnace temperature": "1065 C",
            "Maximum practical cycle time": "24 h",
            "Cooling condition": "controlled furnace cooling",
        },
        top_row={
            "ht_class": "ST_DA",
            "selected_recipe_summary": "980 C for 1 h; 720 C for 8 h; 620 C for 8 h",
            "temperature_time_window": "Solution treatment about 980-1095 C for 1-2 h, then ageing about 720 C/8 h and 620 C/8 h",
            "recommended_peak_temperature_C": 980,
            "recommended_total_hold_h": 17.0,
            "ml_assisted_score": 0.81,
            "confidence": "medium",
            "local_feasibility": "feasible under selected constraints",
        },
        context=ManualInputContext(
            furnace_limit_C=1065,
            maximum_cycle_hours=24,
            section_size="thin section",
            surface_condition="machined",
            build_orientation="vertical",
            initial_material_state="EOS-like LPBF, machined",
            cooling_condition="controlled furnace cooling",
            target_life_cycles=1000000,
            stress_ratio_R=0.1,
        ),
        fatigue_schedule=schedule,
        sn_status={
            "sn_model_trained": True,
            "reviewed_point_rows": 38,
            "registered_targets": 21,
            "status_message": "S-N point review has reached the minimum fitting threshold.",
            "report_note": "Not a design allowable.",
        },
        experiments=build_must_have_experiments("ST_DA", allow_hip=False),
    )

    required = [
        "Technician heat-treatment instruction sheet",
        "Instruction status: draft work instruction for technician review",
        "Do not begin heat treatment until the required blanks below are completed",
        "Material and specimen identification",
        "Specimen or batch ID: to be completed",
        "Equipment and furnace programme",
        "Furnace ID: to be completed",
        "Nominal thermal programme",
        "Stage interpretation through the profile",
        "The full route is ST_DA: solution treatment is completed first",
        "Solution treatment | Ramp to 980 C, then hold 980 C for 1 h",
        "Double ageing - first hold | After solution treatment, hold 720 C for 8 h",
        "Double ageing - second hold | Then hold 620 C for 8 h",
        "Step | Action | Set point | Hold time | Cooling or transfer note",
        "Hold at temperature | 980 C | 1 h",
        "Hold at temperature | 720 C | 8 h",
        "Hold at temperature | 620 C | 8 h",
        "Final cooling method: controlled furnace cooling",
        "Required process records",
        "Operator sign-off",
    ]
    for phrase in required:
        assert phrase in report


def test_select_recommendation_subset_falls_back_and_reports_out_of_grid_fields():
    rows = pd.DataFrame(
        [
            {"target": "balanced", "allow_hip": False, "confidence_mode": "balanced", "ht_class": "ST_DA", "score": 0.7},
            {"target": "fatigue", "allow_hip": False, "confidence_mode": "conservative", "ht_class": "ST_DA", "score": 0.8},
            {"target": "fatigue", "allow_hip": False, "confidence_mode": "conservative", "ht_class": "DA", "score": 0.6},
        ]
    )

    selected, status = select_recommendation_subset(rows, target="fatigue", allow_hip=False, confidence_mode="balanced")

    assert selected["target"].eq("fatigue").all()
    assert selected["confidence_mode"].eq("conservative").all()
    assert status["exact_match"] is False
    assert "outside the reviewed recommendation grid" in status["selection_note"]
    assert status["out_of_grid_fields"][0]["field"] == "Decision posture"
    assert status["out_of_grid_fields"][0]["selected"] == "balanced"
    assert status["out_of_grid_fields"][0]["available"] == "conservative"


def test_select_recommendation_subset_returns_nonempty_global_fallback():
    rows = pd.DataFrame(
        [
            {"target": "balanced", "allow_hip": False, "confidence_mode": "balanced", "ht_class": "ST_DA", "score": 0.7},
        ]
    )

    selected, status = select_recommendation_subset(rows, target="creep", allow_hip=True, confidence_mode="exploratory")

    assert not selected.empty
    assert status["exact_match"] is False
    assert {item["field"] for item in status["out_of_grid_fields"]} == {"Primary design objective", "HIP benchmark inclusion", "Decision posture"}


def test_example_input_combinations_are_available_for_guidance():
    examples = build_example_input_combinations()

    assert {"scenario", "example_inputs", "expected_route_family", "interpretation"}.issubset(examples.columns)
    assert len(examples) >= 3
    assert examples["example_inputs"].str.contains("Primary objective").all()
