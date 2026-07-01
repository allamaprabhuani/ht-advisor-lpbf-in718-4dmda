from ml_project.ht_advisor.validation import validate_model_measurement_row


def test_validation_accepts_curated_row():
    row = {"include_in_model": "true", "curation_status": "curated_reviewed", "evidence_id": "E1", "value": "1234", "unit": "MPa"}
    assert validate_model_measurement_row(row) == []


def test_validation_rejects_unreviewed_row():
    row = {"include_in_model": "true", "curation_status": "candidate", "evidence_id": "E1", "value": "1234", "unit": "MPa"}
    errors = validate_model_measurement_row(row)
    assert "curation_status must be curated_reviewed" in errors

