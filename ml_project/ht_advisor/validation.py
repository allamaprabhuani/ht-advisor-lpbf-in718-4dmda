from __future__ import annotations


def validate_model_measurement_row(row: dict[str, str]) -> list[str]:
    errors = []
    if row.get("include_in_model", "true").lower() != "true":
        errors.append("include_in_model must be true")
    if row.get("curation_status") != "curated_reviewed":
        errors.append("curation_status must be curated_reviewed")
    if not row.get("evidence_id"):
        errors.append("evidence_id is required")
    if not row.get("value"):
        errors.append("value is required")
    if not row.get("unit"):
        errors.append("unit is required")
    return errors

