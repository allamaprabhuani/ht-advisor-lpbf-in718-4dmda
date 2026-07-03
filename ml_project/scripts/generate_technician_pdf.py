from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_project.ht_advisor.dashboard_data import build_thermal_cycle_segment_rows, build_thermal_cycle_rows
from ml_project.ht_advisor.expert_system import (
    ManualInputContext,
    apply_manual_inputs,
    build_fatigue_validation_schedule,
    build_must_have_experiments,
    build_sn_training_status,
)
from ml_project.ht_advisor.recommender import RecommendationRequest, rank_heat_treatments

OUTPUT_PATH = ROOT / "output" / "pdf" / "ht_advisor_fatigue_objective_technician_sheet.pdf"

THERMAL_STEP_COLORS = {
    "Ramp to homogenisation": "#5f6f52",
    "Homogenisation hold": "#8a6f3d",
    "Transition to solution treatment": "#2f5d62",
    "Ramp to solution treatment": "#2f5d62",
    "Solution treatment hold": "#1f7a63",
    "Transition to first ageing": "#5b7f95",
    "First ageing hold": "#a6761d",
    "Transition to second ageing": "#6b7280",
    "Second ageing hold": "#a44a3f",
    "Final cooling": "#374151",
}


def default_context() -> ManualInputContext:
    return ManualInputContext(
        initial_material_state="EOS-like LPBF, as-built",
        build_orientation="vertical",
        section_size="thin coupon",
        surface_condition="machined",
        furnace_limit_C=1100,
        maximum_cycle_hours=20.0,
        cooling_condition="controlled furnace cooling",
        target_life_cycles=1000000,
        stress_ratio_R=0.1,
    )


def ranked_fatigue_route(context: ManualInputContext) -> pd.Series:
    ranked = pd.DataFrame(rank_heat_treatments(RecommendationRequest(target="fatigue", allow_hip=False)))
    adjusted = apply_manual_inputs(ranked, context)
    return adjusted.sort_values("adjusted_rank").iloc[0]


def _style_sheet() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name="Note", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#374151")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=8, spaceAfter=6))
    styles["Title"].fontSize = 18
    styles["Title"].leading = 22
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 12
    return styles


def _table(data: list[list[object]], col_widths: list[float] | None = None, font_size: float = 8.5) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5d6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8faf9")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def draw_thermal_profile(route: str, recipe: str, width: float = 170 * mm, height: float = 82 * mm) -> Drawing:
    segment_rows = build_thermal_cycle_segment_rows(route, recipe)
    cycle_rows = build_thermal_cycle_rows(route, recipe)
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 8, "Thermal step profile: temperature T (C)", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor("#111827")))
    if segment_rows.empty:
        drawing.add(String(0, height / 2, "Thermal-cycle profile unavailable.", fontSize=8, fillColor=colors.HexColor("#6b7280")))
        return drawing

    left, bottom = 32, 30
    plot_w, plot_h = width * 0.55, height - 46
    max_time = max(float(cycle_rows["elapsed_h"].max()), 1.0)
    max_temp = max(float(cycle_rows["temperature_C"].max()), 1000.0)

    def x_pos(elapsed_h: float) -> float:
        return left + (elapsed_h / max_time) * plot_w

    def y_pos(temperature_C: float) -> float:
        return bottom + (temperature_C / max_temp) * plot_h

    drawing.add(Line(left, bottom, left + plot_w, bottom, strokeColor=colors.HexColor("#6b7280"), strokeWidth=0.6))
    drawing.add(Line(left, bottom, left, bottom + plot_h, strokeColor=colors.HexColor("#6b7280"), strokeWidth=0.6))
    for temp in [0, 250, 500, 750, 1000]:
        if temp <= max_temp:
            y = y_pos(temp)
            drawing.add(Line(left - 3, y, left + plot_w, y, strokeColor=colors.HexColor("#e5e7eb"), strokeWidth=0.3))
    for hour in [0, 5, 10, 15, 20]:
        if hour <= max_time:
            x = x_pos(hour)
            drawing.add(Line(x, bottom - 3, x, bottom + plot_h, strokeColor=colors.HexColor("#e5e7eb"), strokeWidth=0.3))
            drawing.add(String(x - 4, bottom - 13, str(hour), fontSize=6.5, fillColor=colors.HexColor("#4b5563")))
    drawing.add(String(left + plot_w / 2 - 18, 5, "Time, t (h)", fontSize=7.5, fillColor=colors.HexColor("#374151")))

    legend_labels: list[str] = []
    for _, segment in segment_rows.groupby("segment_id", sort=True):
        label = str(segment["segment_label"].iloc[0])
        color = HexColor(THERMAL_STEP_COLORS.get(label, "#2f5d62"))
        points = segment.sort_values("elapsed_h")
        x1, y1 = x_pos(float(points.iloc[0]["elapsed_h"])), y_pos(float(points.iloc[0]["temperature_C"]))
        x2, y2 = x_pos(float(points.iloc[1]["elapsed_h"])), y_pos(float(points.iloc[1]["temperature_C"]))
        drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.8))
        drawing.add(Circle(x1, y1, 1.5, fillColor=color, strokeColor=color))
        drawing.add(Circle(x2, y2, 1.5, fillColor=color, strokeColor=color))
        if str(points.iloc[0]["segment_type"]) == "hold" and abs(y2 - y1) < 0.01:
            temperature_label = f"{float(points.iloc[1]['temperature_C']):.0f} C"
            label_x = (x1 + x2) / 2 - 9
            label_y = y1 + 4
            drawing.add(Rect(label_x - 2, label_y - 2, 25, 9, fillColor=colors.white, strokeColor=colors.white))
            drawing.add(String(label_x, label_y, temperature_label, fontSize=6.8, fillColor=color))
        if label not in legend_labels:
            legend_labels.append(label)

    legend_x = left + plot_w + 18
    legend_y = bottom + plot_h - 3
    drawing.add(String(legend_x, legend_y + 12, "Thermal step", fontName="Helvetica-Bold", fontSize=7.5, fillColor=colors.HexColor("#111827")))
    for idx, label in enumerate(legend_labels):
        y = legend_y - idx * 9
        color = HexColor(THERMAL_STEP_COLORS.get(label, "#2f5d62"))
        drawing.add(Rect(legend_x, y - 3, 6, 6, fillColor=color, strokeColor=color))
        drawing.add(String(legend_x + 9, y - 2, label, fontSize=6.6, fillColor=colors.HexColor("#111827")))
    return drawing


def _input_rows(context: ManualInputContext) -> list[list[str]]:
    return [
        ["Field", "Value"],
        ["Primary objective", "fatigue"],
        ["Alloy and process", "LPBF Inconel 718"],
        ["Initial material state", context.initial_material_state],
        ["Build orientation", context.build_orientation],
        ["Surface condition", context.surface_condition],
        ["Representative section size", context.section_size],
        ["HIP benchmark included", "No"],
        ["Maximum furnace temperature", f"{context.furnace_limit_C} C"],
        ["Maximum practical cycle time", f"{context.maximum_cycle_hours:g} h"],
        ["Fatigue validation target", f"R = {context.stress_ratio_R:g}; Nf = {context.target_life_cycles:,} cycles"],
    ]


def build_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    context = default_context()
    top_row = ranked_fatigue_route(context)
    route = str(top_row["ht_class"])
    recipe = str(top_row.get("selected_recipe_summary", top_row.get("temperature_time_window", "")))
    schedule = build_fatigue_validation_schedule(context.stress_ratio_R, context.target_life_cycles)
    experiments = build_must_have_experiments(route, allow_hip=False)
    sn_status = build_sn_training_status(pd.DataFrame())
    styles = _style_sheet()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="HT-Advisor fatigue objective technician sheet",
    )
    story: list[object] = []
    story.append(Paragraph("HT-Advisor technician heat-treatment sheet", styles["Title"]))
    story.append(Paragraph("Fatigue objective - LPBF Inconel 718 - non-HIP route", styles["Note"]))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Input conditions", styles["Section"]))
    story.append(_table(_input_rows(context), [55 * mm, 105 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Recommended route", styles["Section"]))
    route_rows = [
        ["Field", "Value"],
        ["Route", route],
        ["Approved recipe", recipe],
        ["Peak temperature", f"{int(top_row['recommended_peak_temperature_C'])} C"],
        ["Total specified hold time", f"{float(top_row['recommended_total_hold_h']):.1f} h"],
        ["Recommendation index", f"{float(top_row['adjusted_score']):.2f}"],
        ["Local feasibility", str(top_row["local_feasibility"])],
    ]
    story.append(_table(route_rows, [55 * mm, 105 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Nominal thermal programme", styles["Section"]))
    cycle_rows = build_thermal_cycle_rows(route, recipe)
    hold_rows = [["Step", "Set point", "Hold time", "Action"]]
    step_no = 1
    for _, row in cycle_rows.iterrows():
        stage = str(row["stage"])
        if stage.startswith("hold"):
            prior = cycle_rows.iloc[max(int(row.name) - 1, 0)]
            hold_h = float(row["elapsed_h"]) - float(prior["elapsed_h"])
            hold_rows.append([str(step_no), f"{int(row['temperature_C'])} C", f"{hold_h:.1f} h", "Hold at temperature"])
            step_no += 1
    story.append(_table(hold_rows, [15 * mm, 28 * mm, 24 * mm, 93 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(draw_thermal_profile(route, recipe))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "The thermal profile is nominal. Record actual ramp rate, soak start and end times, cooling condition, specimen placement, and thermocouple or witness-coupon position.",
            styles["Note"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Fatigue validation schedule", styles["Section"]))
    schedule_rows = [["sigma_a (MPa)", "R", "sigma_max (MPa)", "sigma_min (MPa)", "Target runout"]]
    for _, item in schedule.iterrows():
        schedule_rows.append(
            [
                f"{int(item['stress_amplitude_MPa'])}",
                f"{float(item['stress_ratio_R']):g}",
                f"{int(item['sigma_max_MPa'])}",
                f"{int(item['sigma_min_MPa'])}",
                str(item["target_runout_cycles"]),
            ]
        )
    story.append(_table(schedule_rows, [26 * mm, 15 * mm, 32 * mm, 32 * mm, 45 * mm], font_size=8))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Required process records", styles["Section"]))
    records = [
        "Furnace ID: to be completed.",
        "Furnace programme ID: to be completed.",
        "Furnace atmosphere, vacuum, or shielding gas: to be completed.",
        "Thermocouple or witness coupon location: to be completed.",
        "Final cooling method: controlled furnace cooling unless locally approved otherwise.",
        "Operator sign-off and process-owner approval: to be completed.",
    ]
    for item in records:
        story.append(Paragraph(f"- {item}", styles["BodyText"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Must-have experimental validation", styles["Section"]))
    for item in experiments:
        story.append(Paragraph(f"- {item['priority']}: {item['experiment']} - {item['reason']}", styles["Small"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("S-N model boundary", styles["Section"]))
    story.append(Paragraph(str(sn_status["report_note"]), styles["Note"]))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    print(build_pdf())
