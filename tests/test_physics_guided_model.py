import json

import pandas as pd

from ml_project.ht_advisor.physics_guided_model import (
    FEATURE_COLUMNS,
    apply_ml_property_ranking,
    build_help_sections,
    build_training_table,
    fit_physics_guided_models,
    predict_candidate_routes,
    train_and_write_artifacts,
)


def test_build_training_table_uses_actual_property_rows_and_physics_features():
    seed = pd.DataFrame(
        [
            {
                "source_row": 1,
                "ref_id": "[1]",
                "reference_url": "https://example.org/one",
                "ht_class": "ST+DA",
                "has_HIP": 0,
                "has_ST": 1,
                "has_DA": 1,
                "solution_temps_C": "980",
                "ageing_temps_C": "720;620",
                "all_times_h": "1;8;8",
                "UTS_MPa": "1450",
                "YS_MPa": "1200",
                "elongation_pct": "8.5",
            },
            {
                "source_row": 2,
                "ref_id": "[2]",
                "reference_url": "https://example.org/two",
                "ht_class": "HIP+ST+DA",
                "has_HIP": 1,
                "has_ST": 1,
                "has_DA": 1,
                "solution_temps_C": "1163",
                "ageing_temps_C": "718;621",
                "all_times_h": "3;8;18",
                "UTS_MPa": "1365",
                "YS_MPa": "1034",
                "elongation_pct": "28.7",
            },
        ]
    )

    table = build_training_table(seed)

    assert table.shape[0] == 2
    assert set(FEATURE_COLUMNS).issubset(table.columns)
    assert table.loc[0, "max_solution_temperature_C"] == 980
    assert table.loc[1, "thermal_activation_index"] > table.loc[0, "thermal_activation_index"]
    assert table["reference_url"].str.startswith("https://").all()


def test_fit_physics_guided_models_calibrates_empirical_models_for_supported_targets():
    training = pd.DataFrame(
        [
            {"has_HIP": 0, "has_ST": 1, "has_DA": 1, "max_solution_temperature_C": 980, "mean_ageing_temperature_C": 670, "total_time_h": 17, "solution_larson_miller": 25000, "ageing_larson_miller": 36000, "thermal_activation_index": 1.2, "UTS_MPa": 1458, "YS_MPa": 1221},
            {"has_HIP": 1, "has_ST": 1, "has_DA": 1, "max_solution_temperature_C": 1163, "mean_ageing_temperature_C": 669.5, "total_time_h": 29, "solution_larson_miller": 29000, "ageing_larson_miller": 39000, "thermal_activation_index": 2.0, "UTS_MPa": 1365, "YS_MPa": 1034},
            {"has_HIP": 0, "has_ST": 1, "has_DA": 1, "max_solution_temperature_C": 1200, "mean_ageing_temperature_C": 670, "total_time_h": 18, "solution_larson_miller": 31000, "ageing_larson_miller": 36000, "thermal_activation_index": 2.4, "UTS_MPa": 1387, "YS_MPa": 1076},
            {"has_HIP": 0, "has_ST": 1, "has_DA": 1, "max_solution_temperature_C": 0, "mean_ageing_temperature_C": 0, "total_time_h": 0, "solution_larson_miller": 0, "ageing_larson_miller": 0, "thermal_activation_index": 0, "UTS_MPa": 1434, "YS_MPa": 1190},
        ]
    )

    report = fit_physics_guided_models(training)

    assert report["model_status"] == "trained"
    assert report["model_family"] == "empirically calibrated parametric model"
    assert report["is_physics_informed_neural_network"] is False
    assert "n = 4" in report["scope_statement"]
    assert "does not explicitly resolve delta-phase fraction" in report["scope_statement"]
    assert "UTS_MPa" in report["trained_targets"]
    assert report["target_models"]["UTS_MPa"]["training_rows"] == 4
    assert "sigma_a = sigma_f_prime * (2Nf)^b" in report["physics_equations"]["basquin_law"]
    assert "feature_ranges" in report


def test_predict_candidate_routes_returns_route_level_ml_property_estimates():
    training = pd.DataFrame(
        [
            {"has_HIP": 0, "has_ST": 1, "has_DA": 1, "max_solution_temperature_C": 980, "mean_ageing_temperature_C": 670, "total_time_h": 17, "solution_larson_miller": 25000, "ageing_larson_miller": 36000, "thermal_activation_index": 1.2, "UTS_MPa": 1458, "YS_MPa": 1221},
            {"has_HIP": 1, "has_ST": 1, "has_DA": 1, "max_solution_temperature_C": 1163, "mean_ageing_temperature_C": 669.5, "total_time_h": 29, "solution_larson_miller": 29000, "ageing_larson_miller": 39000, "thermal_activation_index": 2.0, "UTS_MPa": 1365, "YS_MPa": 1034},
            {"has_HIP": 0, "has_ST": 1, "has_DA": 1, "max_solution_temperature_C": 1200, "mean_ageing_temperature_C": 670, "total_time_h": 18, "solution_larson_miller": 31000, "ageing_larson_miller": 36000, "thermal_activation_index": 2.4, "UTS_MPa": 1387, "YS_MPa": 1076},
        ]
    )
    report = fit_physics_guided_models(training)

    predictions = predict_candidate_routes(report)

    assert {
        "ht_class",
        "predicted_UTS_MPa",
        "predicted_UTS_MPa_lower",
        "predicted_UTS_MPa_upper",
        "empirical_error_bound_note",
        "ml_model_status",
        "outside_training_envelope",
        "training_envelope_note",
    }.issubset(predictions.columns)
    assert "ST_DA" in predictions["ht_class"].tolist()
    assert predictions["predicted_UTS_MPa"].notna().any()
    assert predictions["ml_model_status"].eq("trained").all()
    assert predictions["predicted_UTS_MPa"].between(1365, 1458).all()
    assert (predictions["predicted_UTS_MPa_lower"] <= predictions["predicted_UTS_MPa"]).all()
    assert (predictions["predicted_UTS_MPa_upper"] >= predictions["predicted_UTS_MPa"]).all()
    assert predictions["training_envelope_note"].str.len().gt(0).all()
    assert predictions["empirical_error_bound_note"].str.contains("Empirical error bounds").all()


def test_apply_ml_property_ranking_blends_model_estimates_for_strength_target():
    recommendations = pd.DataFrame(
        [
            {"ht_class": "A", "adjusted_score": 0.70, "adjusted_rank": 1},
            {"ht_class": "B", "adjusted_score": 0.69, "adjusted_rank": 2},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"ht_class": "A", "predicted_UTS_MPa": 1360, "predicted_YS_MPa": 1000},
            {"ht_class": "B", "predicted_UTS_MPa": 1460, "predicted_YS_MPa": 1220},
        ]
    )

    ranked = apply_ml_property_ranking(recommendations, predictions, "strength")

    assert ranked.iloc[0]["ht_class"] == "B"
    assert "ml_property_index" in ranked.columns
    assert "ml_assisted_score" in ranked.columns
    assert ranked.iloc[0]["ml_assisted_rank"] == 1


def test_train_and_write_artifacts_creates_traceable_json_and_predictions(tmp_path):
    seed = pd.DataFrame(
        [
            {"source_row": 1, "ref_id": "[1]", "reference_url": "https://example.org/one", "ht_class": "ST+DA", "has_HIP": 0, "has_ST": 1, "has_DA": 1, "solution_temps_C": "980", "ageing_temps_C": "720;620", "all_times_h": "1;8;8", "UTS_MPa": "1458", "YS_MPa": "1221", "elongation_pct": "6.2"},
            {"source_row": 2, "ref_id": "[2]", "reference_url": "https://example.org/two", "ht_class": "HIP+ST+DA", "has_HIP": 1, "has_ST": 1, "has_DA": 1, "solution_temps_C": "1163", "ageing_temps_C": "718;621", "all_times_h": "3;8;18", "UTS_MPa": "1365", "YS_MPa": "1034", "elongation_pct": "28.7"},
            {"source_row": 3, "ref_id": "[3]", "reference_url": "https://example.org/three", "ht_class": "ST+DA", "has_HIP": 0, "has_ST": 1, "has_DA": 1, "solution_temps_C": "1200", "ageing_temps_C": "720;620", "all_times_h": "2;8;8", "UTS_MPa": "1387", "YS_MPa": "1076", "elongation_pct": "7.5"},
        ]
    )

    report, predictions = train_and_write_artifacts(seed, tmp_path)

    model_json = tmp_path / "physics_guided_model.json"
    predictions_csv = tmp_path / "route_property_predictions.csv"
    training_csv = tmp_path / "physics_guided_training_table.csv"
    assert model_json.exists()
    assert predictions_csv.exists()
    assert training_csv.exists()
    assert json.loads(model_json.read_text())["model_status"] == "trained"
    assert not predictions.empty
    assert predictions_csv.read_text().startswith("ht_class")


def test_help_sections_disclose_how_to_use_and_model_limitations():
    sections = build_help_sections()
    full_text = "\n".join(section["body"] for section in sections)

    assert any(section["title"] == "How to use the tool" for section in sections)
    assert "not a physics-informed neural network" in full_text
    assert "local experimental validation" in full_text
    assert "Empirical error bounds" in full_text
