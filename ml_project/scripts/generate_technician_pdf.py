from __future__ import annotations

import math
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_project.ht_advisor.dashboard_data import build_thermal_cycle_segment_rows, build_thermal_cycle_rows
from ml_project.ht_advisor.expert_system import (
    ManualInputContext,
    apply_manual_inputs,
    build_fatigue_validation_schedule,
    build_must_have_experiments,
)
from ml_project.ht_advisor.recommender import RecommendationRequest, rank_heat_treatments

OUTPUT_PATH = ROOT / "output" / "pdf" / "ht_advisor_fatigue_objective_technician_sheet.pdf"
SN_POINTS_PATH = ROOT / "ml_project" / "data" / "sn_curve_points.csv"

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


def _format_cell(value: object, style: ParagraphStyle, header: bool = False) -> object:
    if isinstance(value, str):
        text = escape(value)
        if header:
            text = f"<b>{text}</b>"
        return Paragraph(text, style)
    return value


def _table(data: list[list[object]], col_widths: list[float] | None = None, font_size: float = 8.5) -> Table:
    cell_style = ParagraphStyle(name=f"TableCell{font_size}", fontSize=font_size, leading=font_size + 2)
    wrapped_data = [
        [_format_cell(value, cell_style, header=(row_idx == 0)) for value in row]
        for row_idx, row in enumerate(data)
    ]
    table = Table(wrapped_data, colWidths=col_widths, repeatRows=1)
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


def _bullet(story: list[object], text: str, style: ParagraphStyle) -> None:
    story.append(Paragraph(f"- {text}", style))


def _goodman_equivalent_amplitude(stress_amplitude_MPa: float, sigma_mean_MPa: float, uts_MPa: float = 1300.0) -> int:
    if sigma_mean_MPa >= uts_MPa:
        return int(round(stress_amplitude_MPa))
    return int(round(stress_amplitude_MPa / (1.0 - sigma_mean_MPa / uts_MPa)))


def _reference_life_rows() -> list[list[str]]:
    return [
        ["Stress amplitude", "Closest ST+DA-like literature context", "Use in this work"],
        ["300 MPa", "Wang et al. (2022) report 8.41e7 cycles for HT-1 at R = -1.", "Runout candidate at R = 0.1; test first."],
        ["350 MPa", "Digitised HT-1 marker: 1.36e7 cycles at R = -1.", "Boundary high-cycle validation level."],
        [
            "400 MPa",
            "Digitised HT-1 markers: 6.85e6 to 8.63e6 cycles at R = -1.",
            "High-stress finite-life screen; do not assume runout.",
        ],
        [
            "450 MPa",
            "Interpolated HT-1 context: approximately 3.0e6 cycles at R = -1.",
            "Optional high-stress finite-life screen after lower levels.",
        ],
    ]


def _test_order_rows(schedule: pd.DataFrame) -> list[list[str]]:
    guidance = {
        300: "Primary runout candidate; perform first to check whether the selected recipe and specimen condition are viable.",
        350: "Second validation level; continue only if 300 MPa reaches runout or gives scientifically useful finite life.",
        400: "High-stress finite-life screen; use to populate the S-N curve if specimens and machine time allow.",
        450: "Highest-stress screen; run last and expect possible early fracture.",
    }
    rows = [["Order", "sigma_a", "sigma_max", "sigma_min", "sigma_mean", "Instruction"]]
    for order, amplitude in enumerate([300, 350, 400, 450], start=1):
        item = schedule[schedule["stress_amplitude_MPa"].eq(amplitude)].iloc[0]
        rows.append(
            [
                str(order),
                f"{amplitude} MPa",
                f"{int(item['sigma_max_MPa'])} MPa",
                f"{int(item['sigma_min_MPa'])} MPa",
                f"{int(item['sigma_mean_MPa'])} MPa",
                guidance[amplitude],
            ]
        )
    return rows


def _schedule_lookup(schedule: pd.DataFrame, stress_amplitude_MPa: int) -> pd.Series:
    match = schedule[schedule["stress_amplitude_MPa"].eq(stress_amplitude_MPa)]
    if match.empty:
        raise ValueError(f"Stress amplitude {stress_amplitude_MPa} MPa is not present in the validation schedule.")
    return match.iloc[0]


def _three_specimen_base_rows(schedule: pd.DataFrame) -> list[list[str]]:
    rows = [["Specimen", "sigma_a", "R", "sigma_max", "sigma_min", "Purpose"]]
    purposes = {
        300: "Primary runout candidate at 1,000,000 cycles.",
        350: "Boundary validation level after the 300 MPa result.",
        400: "Finite-life point if 300 and 350 MPa both reach runout.",
    }
    for specimen_id, stress in enumerate([300, 350, 400], start=1):
        item = _schedule_lookup(schedule, stress)
        rows.append(
            [
                str(specimen_id),
                f"{stress} MPa",
                f"{float(item['stress_ratio_R']):g}",
                f"{int(item['sigma_max_MPa'])} MPa",
                f"{int(item['sigma_min_MPa'])} MPa",
                purposes[stress],
            ]
        )
    return rows


def _three_specimen_adaptive_rows() -> list[list[str]]:
    return [
        ["Observed result", "Next action"],
        ["300 MPa reaches runout", "Test 350 MPa next."],
        ["300 MPa fractures well below target", "Pause testing. Review surface finish, alignment, porosity evidence, and heat-treatment record before using the remaining specimens."],
        ["350 MPa reaches runout", "Use the final specimen at 400 MPa."],
        ["350 MPa fractures before runout", "Use the final specimen at 325 MPa to bracket the transition near 1,000,000 cycles."],
        ["300 and 350 MPa both reach runout", "Use 400 MPa to obtain a finite-life point for the local S-N trend."],
    ]


def _new10_ht1_reference_points() -> pd.DataFrame:
    fallback = pd.DataFrame(
        [
            {"cycles_to_failure": 8.41e7, "stress_amplitude_MPa": 300.0},
            {"cycles_to_failure": 1.3632595e7, "stress_amplitude_MPa": 349.17},
            {"cycles_to_failure": 8.627676e6, "stress_amplitude_MPa": 400.09},
            {"cycles_to_failure": 6.849751e6, "stress_amplitude_MPa": 400.18},
            {"cycles_to_failure": 3.204403e6, "stress_amplitude_MPa": 500.0},
            {"cycles_to_failure": 1.302579e6, "stress_amplitude_MPa": 499.61},
            {"cycles_to_failure": 8.11444e5, "stress_amplitude_MPa": 600.47},
        ]
    )
    if not SN_POINTS_PATH.exists():
        return fallback
    try:
        points = pd.read_csv(SN_POINTS_PATH)
    except Exception:
        return fallback
    required = {"source_id", "heat_treatment_class", "cycles_to_failure", "stress_amplitude_MPa"}
    if not required.issubset(points.columns):
        return fallback
    filtered = points[
        points["source_id"].astype(str).eq("NEW10")
        & points["heat_treatment_class"].astype(str).eq("HT-1")
        & pd.to_numeric(points["cycles_to_failure"], errors="coerce").gt(0)
        & pd.to_numeric(points["stress_amplitude_MPa"], errors="coerce").gt(0)
    ].copy()
    if filtered.empty:
        return fallback
    filtered["cycles_to_failure"] = pd.to_numeric(filtered["cycles_to_failure"], errors="coerce")
    filtered["stress_amplitude_MPa"] = pd.to_numeric(filtered["stress_amplitude_MPa"], errors="coerce")
    return filtered[["cycles_to_failure", "stress_amplitude_MPa"]].dropna()


def _fit_reference_line(points: pd.DataFrame) -> tuple[float, float]:
    x_values = [math.log10(float(value)) for value in points["cycles_to_failure"]]
    y_values = [float(value) for value in points["stress_amplitude_MPa"]]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        return 0.0, y_mean
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def draw_sn_reference_curve(schedule: pd.DataFrame, width: float = 170 * mm, height: float = 105 * mm) -> Drawing:
    drawing = Drawing(width, height)
    drawing.add(
        String(
            0,
            height - 8,
            "Reference markers and planned validation levels",
            fontName="Helvetica-Bold",
            fontSize=9,
            fillColor=colors.HexColor("#111827"),
        )
    )
    drawing.add(
        String(
            0,
            height - 20,
            "Reference HT-1 data are from Wang et al. (2022), R = -1; proposed test levels are R = 0.1.",
            fontSize=7.2,
            fillColor=colors.HexColor("#374151"),
        )
    )

    points = _new10_ht1_reference_points()
    left, bottom = 42, 30
    plot_w, plot_h = width * 0.62, height - 62
    x_min, x_max = 1e5, 1e9
    y_min, y_max = 250.0, 750.0

    def x_pos(cycles: float) -> float:
        return left + (math.log10(cycles) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min)) * plot_w

    def y_pos(stress: float) -> float:
        return bottom + (stress - y_min) / (y_max - y_min) * plot_h

    axis_color = colors.HexColor("#6b7280")
    grid_color = colors.HexColor("#e5e7eb")
    drawing.add(Line(left, bottom, left + plot_w, bottom, strokeColor=axis_color, strokeWidth=0.7))
    drawing.add(Line(left, bottom, left, bottom + plot_h, strokeColor=axis_color, strokeWidth=0.7))

    for cycles in [1e5, 1e6, 1e7, 1e8, 1e9]:
        x = x_pos(cycles)
        drawing.add(Line(x, bottom, x, bottom + plot_h, strokeColor=grid_color, strokeWidth=0.35))
        drawing.add(String(x - 7, bottom - 12, f"1e{int(math.log10(cycles))}", fontSize=6.4, fillColor=colors.HexColor("#4b5563")))
    for stress in [300, 400, 500, 600, 700]:
        y = y_pos(stress)
        drawing.add(Line(left - 3, y, left + plot_w, y, strokeColor=grid_color, strokeWidth=0.35))
        drawing.add(String(left - 26, y - 2, str(stress), fontSize=6.4, fillColor=colors.HexColor("#4b5563")))

    level_color = colors.HexColor("#9a6a2f")
    for _, item in schedule.iterrows():
        stress = float(item["stress_amplitude_MPa"])
        if y_min <= stress <= y_max:
            y = y_pos(stress)
            drawing.add(Line(left, y, left + plot_w, y, strokeColor=level_color, strokeWidth=0.45, strokeDashArray=[2, 2]))
            drawing.add(String(left + plot_w + 4, y - 2, f"{int(stress)} MPa", fontSize=6.2, fillColor=level_color))

    target_x = x_pos(1e6)
    drawing.add(Line(target_x, bottom, target_x, bottom + plot_h, strokeColor=colors.HexColor("#374151"), strokeWidth=0.6, strokeDashArray=[3, 2]))
    drawing.add(String(target_x + 3, bottom + plot_h - 8, "1e6 cycles", fontSize=6.4, fillColor=colors.HexColor("#374151")))

    if len(points) >= 2:
        slope, intercept = _fit_reference_line(points)
        line_color = colors.HexColor("#2f5d62")
        min_cycles = max(x_min, float(points["cycles_to_failure"].min()))
        max_cycles = min(x_max, float(points["cycles_to_failure"].max()))
        line_cycles = [10 ** (math.log10(min_cycles) + idx / 24 * (math.log10(max_cycles) - math.log10(min_cycles))) for idx in range(25)]
        line_points = [
            (x_pos(cycles), y_pos(stress))
            for cycles in line_cycles
            if y_min <= (stress := slope * math.log10(cycles) + intercept) <= y_max
        ]
        for first, second in zip(line_points, line_points[1:]):
            drawing.add(Line(first[0], first[1], second[0], second[1], strokeColor=line_color, strokeWidth=1.4))

    point_color = colors.HexColor("#a44a3f")
    for _, point in points.iterrows():
        cycles = float(point["cycles_to_failure"])
        stress = float(point["stress_amplitude_MPa"])
        if x_min <= cycles <= x_max and y_min <= stress <= y_max:
            drawing.add(Circle(x_pos(cycles), y_pos(stress), 2.1, fillColor=point_color, strokeColor=colors.white, strokeWidth=0.35))

    drawing.add(String(left + plot_w / 2 - 26, 6, "Cycles to failure, Nf", fontSize=7.2, fillColor=colors.HexColor("#374151")))

    legend_x = left + plot_w + 4
    legend_y = bottom + plot_h - 26
    drawing.add(String(legend_x, legend_y + 20, "Legend", fontName="Helvetica-Bold", fontSize=7.2, fillColor=colors.HexColor("#111827")))
    drawing.add(Circle(legend_x + 3, legend_y + 8, 2.1, fillColor=point_color, strokeColor=point_color))
    drawing.add(String(legend_x + 9, legend_y + 6, "HT-1 markers, R = -1", fontSize=6.4, fillColor=colors.HexColor("#111827")))
    drawing.add(Line(legend_x, legend_y - 4, legend_x + 14, legend_y - 4, strokeColor=colors.HexColor("#2f5d62"), strokeWidth=1.4))
    drawing.add(String(legend_x + 18, legend_y - 6, "Reference trend", fontSize=6.4, fillColor=colors.HexColor("#111827")))
    drawing.add(Line(legend_x, legend_y - 16, legend_x + 14, legend_y - 16, strokeColor=level_color, strokeWidth=0.45, strokeDashArray=[2, 2]))
    drawing.add(String(legend_x + 18, legend_y - 18, "Planned levels, R = 0.1", fontSize=6.4, fillColor=colors.HexColor("#111827")))
    return drawing


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
    story.append(PageBreak())
    story.append(Paragraph("Fatigue-test instruction sheet", styles["Title"]))
    story.append(
        Paragraph(
            "Use this page with the local fatigue-machine standard operating procedure. It defines the validation stresses for the recommended heat-treatment route; it does not replace machine-specific safety checks, fixture approval, or laboratory sign-off.",
            styles["Note"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Controller stress targets", styles["Section"]))
    schedule_rows = [
        [
            "sigma_a (MPa)",
            "R",
            "sigma_max (MPa)",
            "sigma_min (MPa)",
            "sigma_mean (MPa)",
            "Target / interpretation",
        ]
    ]
    for _, item in schedule.iterrows():
        amplitude = int(item["stress_amplitude_MPa"])
        if amplitude <= 300:
            interpretation = "Primary runout candidate"
        elif amplitude <= 350:
            interpretation = "Boundary validation level"
        else:
            interpretation = "Finite-life screen; not a runout expectation"
        schedule_rows.append(
            [
                f"{amplitude}",
                f"{float(item['stress_ratio_R']):g}",
                f"{int(item['sigma_max_MPa'])}",
                f"{int(item['sigma_min_MPa'])}",
                f"{int(item['sigma_mean_MPa'])}",
                f"{item['target_runout_cycles']} cycles; {interpretation}",
            ]
        )
    story.append(_table(schedule_rows, [22 * mm, 11 * mm, 25 * mm, 24 * mm, 25 * mm, 53 * mm], font_size=7.8))
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "Stress ratio definition: R = sigma_min / sigma_max. Stress amplitude: sigma_a = (sigma_max - sigma_min) / 2. Mean stress: sigma_mean = (sigma_max + sigma_min) / 2.",
            styles["Note"],
        )
    )
    story.append(
        Paragraph(
            "Force conversion for the machine controller: Fmax = sigma_max x A0 and Fmin = sigma_min x A0, where A0 is the measured net cross-sectional area in mm2. The resulting force is in N when stress is in MPa.",
            styles["Note"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Recommended test sequence and review points", styles["Section"]))
    story.append(_table(_test_order_rows(schedule), [13 * mm, 20 * mm, 23 * mm, 23 * mm, 23 * mm, 58 * mm], font_size=7.4))
    story.append(Spacer(1, 3 * mm))
    _bullet(
        story,
        "Start with 300 MPa. If the specimen fractures far below the target runout, pause and review surface finish, porosity evidence, alignment, and heat-treatment records before consuming higher-stress specimens.",
        styles["Small"],
    )
    _bullet(
        story,
        "Run 400 MPa and 450 MPa only when finite-life data are required for fitting the local S-N curve or when sufficient spare specimens are available.",
        styles["Small"],
    )
    _bullet(
        story,
        "Use the same specimen geometry, surface condition, build orientation, waveform, and frequency across the validation set unless the research lead approves a controlled comparison.",
        styles["Small"],
    )
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Stop criteria", styles["Section"]))
    stop_criteria = [
        "Stop at specimen fracture.",
        "Stop at 1,000,000 cycles and record the result as runout if fracture has not occurred.",
        "Stop and flag the test if grip slip, load-control instability, fixture contact, abnormal displacement drift, machine alarm, or visible specimen heating occurs.",
        "Do not restart a stopped fatigue test on the same specimen unless the research lead records the reason and approves the restart.",
    ]
    for item in stop_criteria:
        _bullet(story, item, styles["Small"])
    story.append(PageBreak())
    story.append(Paragraph("Three-specimen fatigue test reference", styles["Title"]))
    story.append(
        Paragraph(
            "Use this sheet when only three fatigue specimens are available for the ST_DA validation route. The aim is to obtain the highest-value evidence: one likely runout point, one boundary point, and one finite-life point if the first two results permit it.",
            styles["Note"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Initial test plan", styles["Section"]))
    story.append(_table(_three_specimen_base_rows(schedule), [17 * mm, 22 * mm, 11 * mm, 26 * mm, 25 * mm, 59 * mm], font_size=7.8))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Adaptive selection for the final specimen", styles["Section"]))
    story.append(_table(_three_specimen_adaptive_rows(), [58 * mm, 102 * mm], font_size=8.0))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Technician notes", styles["Section"]))
    _bullet(story, "Run the specimens in sequence: 300 MPa first, then 350 MPa if the first result is acceptable.", styles["Small"])
    _bullet(story, "Do not continue to a higher stress level after an unexpectedly early fracture until the research lead has reviewed the specimen and setup records.", styles["Small"])
    _bullet(story, "Keep heat treatment, build orientation, surface condition, geometry, waveform, frequency, and room-temperature test condition unchanged across the three specimens.", styles["Small"])
    _bullet(story, "Record runout and fractured specimens separately. A runout at 1,000,000 cycles is not a measured failure life.", styles["Small"])
    story.append(PageBreak())
    story.append(Paragraph("Test records and validation controls", styles["Title"]))
    story.append(Paragraph("Required heat-treatment records", styles["Section"]))
    records = [
        "Furnace ID: to be completed.",
        "Furnace programme ID: to be completed.",
        "Furnace atmosphere, vacuum, or shielding gas: to be completed.",
        "Thermocouple or witness coupon location: to be completed.",
        "Final cooling method: controlled furnace cooling unless locally approved otherwise.",
        "Operator sign-off and process-owner approval: to be completed.",
    ]
    for item in records:
        _bullet(story, item, styles["BodyText"])
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Required fatigue-test records", styles["Section"]))
    fatigue_records = [
        "Specimen ID, heat-treatment batch ID, build orientation, surface condition, and measured roughness if available.",
        "Measured gauge dimensions and net cross-sectional area A0 used for load conversion.",
        "Machine ID, load cell range, fixture/grip type, alignment check, waveform, frequency, room temperature, and operator name.",
        "Programmed sigma_a, sigma_max, sigma_min, sigma_mean, R, Fmax, Fmin, and target runout.",
        "Actual cycles to fracture or runout, failure location, fracture-surface photographs, and notes on grip slip or abnormal heating.",
    ]
    for item in fatigue_records:
        _bullet(story, item, styles["BodyText"])
    story.append(PageBreak())
    story.append(Paragraph("Illustrative S-N curve context", styles["Title"]))
    story.append(draw_sn_reference_curve(schedule))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "This plot is included to show the expected form of the fatigue dataset and the relative placement of the proposed stress levels. It should not be used to claim fatigue life for the present R = 0.1 tests because the reference markers were obtained at R = -1.",
            styles["Note"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "After testing, replace the planned-level guide lines with the measured local specimen points and report runout specimens separately from fractured specimens.",
            styles["Note"],
        )
    )
    story.append(PageBreak())
    story.append(Paragraph("Literature context and interpretation limits", styles["Title"]))
    story.append(Paragraph("Reference evidence", styles["Section"]))
    story.append(
        Paragraph(
            "Closest reviewed fatigue source: Wang et al. (2022), Journal of Alloys and Compounds 913:165171, DOI 10.1016/j.jallcom.2022.165171, Fig. 14. The paper reports selective laser melted Inconel 718 tested at room temperature using R = -1. Its HT-1 route is ST+DA-like but not identical to the present ST_DA recommendation.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(_table(_reference_life_rows(), [28 * mm, 72 * mm, 60 * mm], font_size=7.6))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Mean-stress caution", styles["Section"]))
    story.append(
        Paragraph(
            "The present validation schedule uses R = 0.1, which introduces tensile mean stress. The Wang et al. values above are therefore reference context only, not fatigue-life predictions for these specimens.",
            styles["BodyText"],
        )
    )
    mean_rows = [["sigma_a", "sigma_mean at R = 0.1", "Approx. Goodman-equivalent R = -1 amplitude", "Interpretation"]]
    for _, item in schedule.iterrows():
        amplitude = int(item["stress_amplitude_MPa"])
        sigma_mean = int(item["sigma_mean_MPa"])
        equivalent = _goodman_equivalent_amplitude(amplitude, sigma_mean)
        interpretation = (
            "possible runout candidate"
            if amplitude == 300
            else "increasing finite-life risk"
            if amplitude == 350
            else "likely finite-life/rapid-failure regime"
        )
        mean_rows.append([f"{amplitude} MPa", f"{sigma_mean} MPa", f"{equivalent} MPa, assuming UTS = 1300 MPa", interpretation])
    story.append(_table(mean_rows, [24 * mm, 34 * mm, 58 * mm, 44 * mm], font_size=7.6))
    story.append(Spacer(1, 3 * mm))
    _bullet(
        story,
        "Treat the Goodman-equivalent values as a screening calculation only. Actual fatigue life can vary strongly with internal porosity, lack-of-fusion defects, surface condition, residual stress, and build orientation.",
        styles["Small"],
    )
    _bullet(
        story,
        "Because the route is non-HIP, scatter between nominally identical specimens should be expected. Report failed specimens and runout specimens separately.",
        styles["Small"],
    )
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Must-have experimental validation", styles["Section"]))
    for item in experiments:
        _bullet(story, f"{item['priority']}: {item['experiment']} - {item['reason']}", styles["Small"])
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("S-N model boundary", styles["Section"]))
    story.append(
        Paragraph(
            "A literature-only right-censored Basquin S-N module has been trained from reviewed marker points at R = -1 or with stress ratio not reported. "
            "Runout markers are retained as right-censored lower-bound observations. "
            "No R = 0.1 local fatigue-life predictor is trained yet. The validation schedule above is intended to generate local fatigue data for calibration; it should not be cited as a trained local fatigue-life prediction.",
            styles["Note"],
        )
    )

    doc.build(story)
    return output_path


if __name__ == "__main__":
    print(build_pdf())
