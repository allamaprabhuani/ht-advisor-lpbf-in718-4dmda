from pathlib import Path

import pandas as pd

from ml_project.ht_advisor.physics_guided_model import train_and_write_artifacts
from ml_project.scripts import train_sn_fatigue_model


ROOT = Path(__file__).resolve().parents[1]


def test_static_and_sn_model_cards_state_data_counts_boundaries_and_allowed_claims():
    static_card = ROOT / "ml_project" / "model_outputs" / "static_model_card.md"
    sn_card = ROOT / "ml_project" / "model_outputs" / "sn_model_card.md"

    assert static_card.exists()
    assert sn_card.exists()

    static_text = static_card.read_text(encoding="utf-8")
    sn_text = sn_card.read_text(encoding="utf-8")

    for phrase in [
        "Formal Model Card",
        "Data Counts",
        "Excluded Data",
        "Assumptions",
        "Limitations",
        "Allowed Claims",
        "Claims Not Supported",
    ]:
        assert phrase in static_text
        assert phrase in sn_text

    assert "static tensile indicators" in static_text
    assert "not a physics-informed neural network" in static_text
    assert "calibration rows" in static_text

    assert "38 reviewed digitised S-N marker points" in sn_text
    assert "R = -1" in sn_text
    assert "not reported" in sn_text
    assert "right-censored runout" in sn_text
    assert "No local R = 0.1 fatigue-life predictor" in sn_text


def test_dashboard_readme_no_longer_says_reviewed_sn_point_table_is_empty():
    readme = ROOT / "ml_project" / "dashboard" / "README.md"
    text = readme.read_text(encoding="utf-8")

    assert "That point table remains empty" not in text
    assert "38 reviewed digitised S-N marker points" in text
    assert "five condition-specific literature Basquin screening fits" in text


def test_model_card_artifacts_are_regenerated_with_model_outputs(tmp_path):
    seed_rows = pd.read_csv(ROOT / "ml_project" / "extracted_data" / "excel_seed_mechanical_dataset.csv")

    train_and_write_artifacts(seed_rows, tmp_path)
    sn_report = train_sn_fatigue_model.train_and_write_sn_artifacts(tmp_path, tmp_path)

    static_card = tmp_path / "static_model_card.md"
    sn_card = tmp_path / "sn_model_card.md"
    assert static_card.exists()
    assert sn_card.exists()
    assert "calibration rows" in static_card.read_text(encoding="utf-8")
    assert str(sn_report["reviewed_points"]) in sn_card.read_text(encoding="utf-8")
