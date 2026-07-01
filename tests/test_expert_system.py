import pandas as pd

from ml_project.ht_advisor.expert_system import (
    ManualInputContext,
    apply_manual_inputs,
    build_must_have_experiments,
    build_model_specification,
    build_raw_training_data_table,
    generate_text_recommendation,
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


def test_text_recommendation_reports_route_effects_and_stochastic_uncertainty():
    row = {
        "ht_class": "ST_DA",
        "temperature_time_window": "Solution treatment about 980-1095 C for 1-2 h, then ageing about 720 C/8 h and 620 C/8 h",
        "local_feasibility": "feasible under selected constraints",
        "constraint_notes": "No constraint penalties were applied.",
        "recommendation_reason": "Literature-supported non-HIP route.",
        "confidence": "medium",
        "adjusted_score": 0.79,
    }
    text = generate_text_recommendation(row, ManualInputContext(target_life_cycles=1000000, surface_condition="machined"))

    assert "ST_DA" in text
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
