from pathlib import Path


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
