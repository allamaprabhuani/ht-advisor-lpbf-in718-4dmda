from ml_project.ht_advisor.literature_evidence import (
    build_recommendation_literature_notes,
    build_supporting_literature_table,
)


def test_supporting_literature_table_contains_song_and_jirandehi_results():
    table = build_supporting_literature_table()

    assert {"citation_key", "doi", "use_in_app", "extracted_result"}.issubset(table.columns)
    assert "Song2025_Materials_ML_Fatigue" in table["citation_key"].tolist()
    assert "Jirandehi2022_AdditiveManufacturing_BuildOrientation" in table["citation_key"].tolist()
    assert "Metals2021_HT_Optimisation" in table["citation_key"].tolist()
    assert "JALCOM2022_HT_TensileFatigue" in table["citation_key"].tolist()
    assert table["doi"].str.contains("10.3390/ma18112604", regex=False).any()
    assert table["doi"].str.contains("10.1016/j.addma.2022.102661", regex=False).any()
    assert table["doi"].str.contains("10.3390/met11010144", regex=False).any()
    assert table["doi"].str.contains("10.1016/j.jallcom.2022.165171", regex=False).any()
    assert table["extracted_result"].str.contains("GAN-RF", regex=False).any()


def test_recommendation_notes_include_ml_fatigue_feature_requirements_for_fatigue_target():
    notes = build_recommendation_literature_notes(target="fatigue", build_orientation="vertical")
    combined = " ".join(note["note"] for note in notes)

    assert "stress amplitude" in combined
    assert "defect" in combined
    assert "build orientation" in combined
    assert "heat-treatment" in combined
    assert "surface condition" in combined
    assert any(note["citation_key"] == "Song2025_Materials_ML_Fatigue" for note in notes)
    assert any(note["citation_key"] == "Jirandehi2022_AdditiveManufacturing_BuildOrientation" for note in notes)
    assert any(note["citation_key"] == "JALCOM2022_HT_TensileFatigue" for note in notes)
    assert any(note["citation_key"] == "Witkin2020_SurfaceOrientationFatigue" for note in notes)


def test_recommendation_notes_include_new_heat_treatment_sources_for_balanced_target():
    notes = build_recommendation_literature_notes(target="balanced", build_orientation="not specified")
    keys = {note["citation_key"] for note in notes}
    combined = " ".join(note["note"] for note in notes)

    assert "Metals2021_HT_Optimisation" in keys
    assert "JALCOM2022_ST_DA_Mechanical" in keys
    assert "Materials2020_HA_ST_Time" in keys
    assert "solution treatment plus ageing" in combined
    assert "HA_ST_DA" in combined


def test_recommendation_notes_do_not_treat_supporting_papers_as_training_rows():
    table = build_supporting_literature_table()

    assert table["model_use"].eq("supporting_literature_not_training_row").all()
