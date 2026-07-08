from pathlib import Path

from reportlab.graphics.shapes import String

from ml_project.ht_advisor.expert_system import build_fatigue_validation_schedule
from ml_project.scripts.generate_technician_pdf import (
    _three_specimen_adaptive_rows,
    _three_specimen_base_rows,
    draw_sn_reference_curve,
    draw_thermal_profile,
)


def test_pdf_generator_uses_segmented_thermal_profile_and_step_legend():
    script_text = Path("ml_project/scripts/generate_technician_pdf.py").read_text(encoding="utf-8")
    required = [
        "build_thermal_cycle_segment_rows",
        "THERMAL_STEP_COLORS",
        "draw_thermal_profile",
        "Thermal step",
        "Ramp to solution treatment",
        "Solution treatment hold",
        "First ageing hold",
        "Second ageing hold",
        "Final cooling",
        "A literature-only right-censored Basquin S-N module has been trained",
        "Runout markers are retained as right-censored lower-bound observations",
        "No R = 0.1 local fatigue-life predictor is trained yet",
    ]
    for phrase in required:
        assert phrase in script_text


def _drawing_strings(item):
    for child in getattr(item, "contents", []):
        if isinstance(child, String):
            yield str(child.text)
        yield from _drawing_strings(child)


def test_pdf_thermal_profile_prints_temperature_values_on_hold_segments_only():
    drawing = draw_thermal_profile("ST_DA", "980 C for 1 h; 720 C for 8 h; 620 C for 8 h")
    labels = list(_drawing_strings(drawing))

    assert "Time, t (h)" in labels
    assert "Thermal step profile: temperature T (C)" in labels
    assert "980 C" in labels
    assert "720 C" in labels
    assert "620 C" in labels
    assert "250" not in labels
    assert "500" not in labels
    assert "750" not in labels
    assert "1000" not in labels


def test_pdf_sn_reference_curve_labels_reference_and_planned_levels():
    schedule = build_fatigue_validation_schedule(stress_ratio_R=0.1, target_life_cycles=1000000)
    drawing = draw_sn_reference_curve(schedule)
    labels = list(_drawing_strings(drawing))

    assert "Reference markers and planned validation levels" in labels
    assert "HT-1 markers, R = -1" in labels
    assert "Planned levels, R = 0.1" in labels
    assert "1e6 cycles" in labels
    assert "300 MPa" in labels
    assert "450 MPa" in labels


def test_pdf_three_specimen_reference_uses_adaptive_final_stress():
    schedule = build_fatigue_validation_schedule(stress_ratio_R=0.1, target_life_cycles=1000000)
    base_rows = _three_specimen_base_rows(schedule)
    adaptive_rows = _three_specimen_adaptive_rows()

    assert base_rows[1][1:] == ["300 MPa", "0.1", "667 MPa", "67 MPa", "Primary runout candidate at 1,000,000 cycles."]
    assert any("Use the final specimen at 325 MPa" in row[1] for row in adaptive_rows)
    assert any("Use the final specimen at 400 MPa" in row[1] for row in adaptive_rows)
