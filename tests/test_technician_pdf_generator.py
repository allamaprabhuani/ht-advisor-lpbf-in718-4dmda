from pathlib import Path

from reportlab.graphics.shapes import String

from ml_project.scripts.generate_technician_pdf import draw_thermal_profile


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
