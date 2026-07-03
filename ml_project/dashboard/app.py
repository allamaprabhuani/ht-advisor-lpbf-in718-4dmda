from __future__ import annotations

import json
from pathlib import Path
import sys
import inspect

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_project.ht_advisor.dashboard_data import (
    build_process_window_rows,
    build_property_tradeoff_rows,
    build_recommendation_contribution_rows,
    build_route_radar_rows,
    build_thermal_cycle_segment_rows,
    build_thermal_cycle_rows,
)
from ml_project.ht_advisor.expert_system import (
    ManualInputContext,
    apply_manual_inputs,
    build_must_have_experiments,
    build_model_specification,
    build_raw_training_data_table,
    generate_text_recommendation,
)
try:
    from ml_project.ht_advisor.expert_system import build_printable_recommendation_report, build_sn_training_status
except ImportError:
    def build_sn_training_status(sn_points: pd.DataFrame, sn_targets: pd.DataFrame | None = None) -> dict[str, object]:
        reviewed_point_rows = 0
        if sn_points is not None and not sn_points.empty:
            if "review_status" in sn_points.columns:
                reviewed_point_rows = int(sn_points["review_status"].astype(str).str.contains("reviewed", case=False, na=False).sum())
            else:
                reviewed_point_rows = int(len(sn_points))
        return {
            "sn_model_trained": False,
            "reviewed_point_rows": reviewed_point_rows,
            "registered_targets": int(len(sn_targets)) if sn_targets is not None else 0,
            "status_message": "S-N curves have not yet been trained because the reviewed point table does not contain enough approved fatigue data.",
            "report_note": "Fatigue life is not predicted in the current release; stress ratio and target cycles are used only to plan validation tests.",
        }

    def build_printable_recommendation_report(
        input_conditions: dict[str, object],
        top_row: dict | pd.Series,
        context: ManualInputContext,
        fatigue_schedule: pd.DataFrame,
        sn_status: dict[str, object],
        experiments: list[dict[str, str]] | None = None,
    ) -> str:
        row = dict(top_row)
        lines = ["# Printable recommendation report", "", "## Process & Material Specifications", "", "### Full input conditions"]
        for key, value in input_conditions.items():
            lines.append(f"- {key}: {value}")
        lines.extend(
            [
                "",
                "## Recommended heat-treatment route",
                "",
                f"- Route: {row.get('ht_class', 'not available')}",
                f"- Proposed validation recipe: {row.get('selected_recipe_summary', row.get('temperature_time_window', 'not specified'))}",
                f"- Recommendation index: {float(row.get('ml_assisted_score', row.get('adjusted_score', 0.0))):.2f}",
                f"- Fatigue validation context: R = {context.stress_ratio_R:g}; Nf = {context.target_life_cycles:,} cycles" if context.target_life_cycles else f"- Fatigue validation context: R = {context.stress_ratio_R:g}; Nf not specified",
                "",
                "## Technician heat-treatment instruction sheet",
                "",
                "- Instruction status: draft work instruction for technician review; verify against the local furnace standard operating procedure before processing.",
                "- Do not begin heat treatment until the required blanks below are completed and the route is approved by the process owner.",
                "",
                "### Material and specimen identification",
                "",
                "- Specimen or batch ID: to be completed",
                f"- Initial material state: {context.initial_material_state}",
                f"- Build orientation: {context.build_orientation}",
                f"- Surface condition: {context.surface_condition}",
                f"- Representative section size: {context.section_size}",
                "",
                "### Equipment and furnace programme",
                "",
                "- Furnace ID: to be completed",
                "- Furnace programme ID: to be completed",
                "- Furnace atmosphere, vacuum, or shielding gas: to be completed",
                f"- Final cooling method: {context.cooling_condition}",
                "",
                "### Nominal thermal programme",
                "",
                "- Use the proposed validation recipe above as the nominal thermal programme.",
                "",
                "### Required process records",
                "",
                "- Record actual ramp rate, soak start time, soak end time, cooling condition, specimen placement, and thermocouple or witness coupon position.",
                "",
                "### Sign-off",
                "",
                "- Operator sign-off: to be completed",
                "",
                "## Expected static-property estimates",
            ]
        )
        for label, column, unit in [
            ("UTS", "predicted_UTS_MPa", "MPa"),
            ("YS", "predicted_YS_MPa", "MPa"),
            ("Elongation", "predicted_elongation_pct", "%"),
        ]:
            if column in row and pd.notna(row.get(column)):
                lines.append(f"- {label}: {float(row[column]):.1f} {unit}")
        lines.extend(["", "## Fatigue validation schedule"])
        for _, item in fatigue_schedule.iterrows():
            lines.append(
                f"- sigma_a = {int(item['stress_amplitude_MPa'])} MPa; sigma_max = {int(item['sigma_max_MPa'])} MPa; "
                f"sigma_min = {int(item['sigma_min_MPa'])} MPa; sigma_mean = {int(item['sigma_mean_MPa'])} MPa; "
                f"target runout = {item['target_runout_cycles']} cycles"
            )
        lines.extend(
            [
                "",
                "## S-N training status",
                f"- Status: {sn_status.get('status_message')}",
                f"- Boundary: {sn_status.get('report_note')}",
                "",
                "## Must-have experimental validation",
            ]
        )
        for item in experiments or []:
            lines.append(f"- {item['priority']}: {item['experiment']} - {item['reason']}")
        return "\n".join(lines)


def build_printable_report_safely(
    input_conditions: dict[str, object],
    top_row: dict | pd.Series,
    context: ManualInputContext,
    fatigue_schedule: pd.DataFrame,
    sn_status: dict[str, object],
    experiments: list[dict[str, str]] | None = None,
) -> str:
    try:
        return build_printable_recommendation_report(
            input_conditions=input_conditions,
            top_row=top_row,
            context=context,
            fatigue_schedule=fatigue_schedule,
            sn_status=sn_status,
            experiments=experiments,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        row = dict(top_row)
        lines = [
            "# Printable recommendation report",
            "",
            "## Process & Material Specifications",
            "",
            "### Full input conditions",
        ]
        for key, value in input_conditions.items():
            lines.append(f"- {key}: {value}")
        lines.extend(
            [
                "",
                "## Recommended heat-treatment route",
                "",
                f"- Route: {row.get('ht_class', 'not available')}",
                f"- Proposed validation recipe: {row.get('selected_recipe_summary', row.get('temperature_time_window', 'not specified'))}",
                f"- Recommendation index: {float(row.get('ml_assisted_score', row.get('adjusted_score', 0.0))):.2f}",
                "",
                "## Technician heat-treatment instruction sheet",
                "",
                "- Instruction status: draft work instruction for technician review; verify against the local furnace standard operating procedure before processing.",
                "- Do not begin heat treatment until the required blanks below are completed and the route is approved by the process owner.",
                "",
                "### Material and specimen identification",
                "",
                "- Specimen or batch ID: to be completed",
                f"- Initial material state: {context.initial_material_state}",
                f"- Build orientation: {context.build_orientation}",
                f"- Surface condition: {context.surface_condition}",
                f"- Representative section size: {context.section_size}",
                "",
                "### Equipment and furnace programme",
                "",
                "- Furnace ID: to be completed",
                "- Furnace programme ID: to be completed",
                "- Furnace atmosphere, vacuum, or shielding gas: to be completed",
                f"- Final cooling method: {context.cooling_condition}",
                "",
                "### Nominal thermal programme",
                "",
                "- Use the proposed validation recipe above as the nominal thermal programme.",
                "",
                "### Required process records",
                "",
                "- Record actual ramp rate, soak start time, soak end time, cooling condition, specimen placement, and thermocouple or witness coupon position.",
                "",
                "### Sign-off",
                "",
                "- Operator sign-off: to be completed",
                "",
                "## Expected static-property estimates",
            ]
        )
        for label, column, unit in [
            ("UTS", "predicted_UTS_MPa", "MPa"),
            ("YS", "predicted_YS_MPa", "MPa"),
            ("Elongation", "predicted_elongation_pct", "%"),
        ]:
            if column in row and pd.notna(row.get(column)):
                lines.append(f"- {label}: {float(row[column]):.1f} {unit}")
        lines.extend(["", "## Fatigue validation schedule"])
        for _, item in fatigue_schedule.iterrows():
            lines.append(
                f"- sigma_a = {int(item['stress_amplitude_MPa'])} MPa; sigma_max = {int(item['sigma_max_MPa'])} MPa; "
                f"sigma_min = {int(item['sigma_min_MPa'])} MPa; sigma_mean = {int(item['sigma_mean_MPa'])} MPa; "
                f"target runout = {item['target_runout_cycles']} cycles"
            )
        lines.extend(
            [
                "",
                "## S-N training status",
                f"- Status: {sn_status.get('status_message')}",
                f"- Boundary: {sn_status.get('report_note')}",
                "",
                "## Must-have experimental validation",
            ]
        )
        for item in experiments or []:
            lines.append(f"- {item['priority']}: {item['experiment']} - {item['reason']}")
        return "\n".join(lines)


def extract_technician_sheet(report_markdown: str) -> str:
    start_heading = "## Technician heat-treatment instruction sheet"
    end_heading = "## Expected static-property estimates"
    start = report_markdown.find(start_heading)
    if start < 0:
        return report_markdown
    end = report_markdown.find(end_heading, start)
    if end < 0:
        return report_markdown[start:].strip()
    return report_markdown[start:end].strip()
try:
    from ml_project.ht_advisor.expert_system import build_fatigue_validation_schedule
except ImportError:
    def build_fatigue_validation_schedule(
        stress_ratio_R: float = 0.1,
        target_life_cycles: int | None = 1000000,
        stress_amplitudes_MPa: list[int] | None = None,
    ) -> pd.DataFrame:
        amplitudes = stress_amplitudes_MPa or [300, 350, 400, 450]
        rows = []
        for amplitude in amplitudes:
            sigma_max = 2.0 * float(amplitude) / (1.0 - float(stress_ratio_R))
            sigma_min = float(stress_ratio_R) * sigma_max
            sigma_mean = 0.5 * (sigma_max + sigma_min)
            rows.append(
                {
                    "stress_amplitude_MPa": int(amplitude),
                    "stress_ratio_R": round(float(stress_ratio_R), 3),
                    "sigma_max_MPa": int(round(sigma_max)),
                    "sigma_min_MPa": int(round(sigma_min)),
                    "sigma_mean_MPa": int(round(sigma_mean)),
                    "target_runout_cycles": int(target_life_cycles) if target_life_cycles else "not specified",
                    "interpretation": "validation stress level, not predicted life",
                }
            )
        return pd.DataFrame(rows)
try:
    from ml_project.ht_advisor.expert_system import build_example_input_combinations, select_recommendation_subset
except ImportError:
    build_example_input_combinations = None
    select_recommendation_subset = None
from ml_project.ht_advisor.literature_evidence import build_recommendation_literature_notes, build_supporting_literature_table
from ml_project.ht_advisor.physics_guided_model import build_help_sections
from ml_project.ht_advisor.physics_guided_model import apply_ml_property_ranking
try:
    from ml_project.ht_advisor.physics_guided_model import build_equation_table, build_notation_table
except ImportError:
    build_equation_table = None
    build_notation_table = None

CURATED = ROOT / "ml_project" / "curated_data"
OUTPUTS = ROOT / "ml_project" / "model_outputs"
EXTRACTED = ROOT / "ml_project" / "extracted_data"
LITERATURE = ROOT / "ml_project" / "literature_search"
DATA = ROOT / "ml_project" / "data"
REPORTS = ROOT / "ml_project" / "reports"

st.set_page_config(page_title="HT-Advisor", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --academic-ink: #17212b;
        --academic-muted: #4b5a66;
        --academic-border: #d8dee4;
        --academic-panel: #f4f6f7;
        --academic-panel-strong: #eef3f2;
        --academic-accent: #22535a;
        --academic-accent-soft: #e6eff0;
        --academic-burgundy: #8d3f3f;
        --academic-gold: #9b7a2f;
    }
    .stApp {
        background: #f8f8f5;
        color: var(--academic-ink);
    }
    [data-testid="stHeader"] {
        background: #f8f8f5;
    }
    [data-testid="stSidebar"] {
        background: #eef3f2;
        border-right: 1px solid #c9d5d4;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #183b42;
    }
    h1, h2, h3 {
        color: var(--academic-ink);
        letter-spacing: 0;
    }
    h1 {
        border-bottom: 3px solid var(--academic-accent);
        padding-bottom: 0.35rem;
    }
    div[data-testid="stMetric"] {
        background: var(--academic-panel);
        border: 1px solid var(--academic-border);
        border-radius: 4px;
        padding: 0.65rem 0.75rem;
        box-shadow: 0 1px 2px rgba(23, 33, 43, 0.05);
    }
    div[data-testid="stInfo"] {
        background: var(--academic-accent-soft);
        border: 1px solid #b7cacc;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--academic-border);
        border-radius: 4px;
    }
    button[kind="secondary"] {
        border-radius: 4px;
    }
    .route-chip {
        display: inline-block;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        background: #e6eff0;
        border: 1px solid #b7cacc;
        color: #183b42;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .context-note {
        padding: 0.75rem 0.85rem;
        border-left: 4px solid var(--academic-accent);
        background: #f4f6f7;
        color: var(--academic-ink);
    }
    .print-report {
        background: #ffffff;
        border: 1px solid var(--academic-border);
        border-radius: 4px;
        padding: 1rem 1.1rem;
    }
    .print-report h3 {
        margin-top: 0.25rem;
    }
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], button {
            display: none !important;
        }
        .stApp {
            background: #ffffff;
        }
        .print-report {
            border: none;
            padding: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("HT-Advisor: Evidence-Guided Heat-Treatment Selection for LPBF Inconel 718")
st.caption(
    "A source-traceable decision-support system for ranking heat-treatment routes under local processing constraints. "
    "Recommendations indicate literature-supported experimental candidates and require validation on local specimens."
)


def file_fingerprint(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


@st.cache_data
def load_csv(path: Path, fingerprint: tuple[int, int]) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_json(path: Path, fingerprint: tuple[int, int]) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


recs_path = OUTPUTS / "ht_recommendations.csv"
sources_path = CURATED / "sources.csv"
source_files_path = CURATED / "source_files.csv"
recipes_path = CURATED / "heat_treatment_recipes.csv"
measurements_path = CURATED / "mechanical_measurements.csv"
scope_path = EXTRACTED / "corpus_scope_audit.csv"
online_manifest_path = LITERATURE / "final_online_source_manifest.csv"
trained_model_path = OUTPUTS / "physics_guided_model.json"
route_predictions_path = OUTPUTS / "route_property_predictions.csv"
training_table_path = OUTPUTS / "physics_guided_training_table.csv"
sn_targets_path = DATA / "sn_digitisation_targets.csv"
sn_points_path = DATA / "sn_curve_points.csv"
sn_review_queue_path = REPORTS / "sn_pdf_review_queue.csv"
sn_audit_summary_path = REPORTS / "sn_digitisation_audit_summary.json"
sn_model_summary_path = OUTPUTS / "sn_model_summary.csv"
sn_model_prediction_grid_path = OUTPUTS / "sn_model_prediction_grid.csv"
sn_model_artifact_path = OUTPUTS / "sn_model_artifact.json"

recs = load_csv(recs_path, file_fingerprint(recs_path))
sources = load_csv(sources_path, file_fingerprint(sources_path))
source_files = load_csv(source_files_path, file_fingerprint(source_files_path))
recipes = load_csv(recipes_path, file_fingerprint(recipes_path))
measurements = load_csv(measurements_path, file_fingerprint(measurements_path))
scope = load_csv(scope_path, file_fingerprint(scope_path))
online_manifest = load_csv(online_manifest_path, file_fingerprint(online_manifest_path))
trained_model = load_json(trained_model_path, file_fingerprint(trained_model_path))
route_predictions = load_csv(route_predictions_path, file_fingerprint(route_predictions_path))
training_table = load_csv(training_table_path, file_fingerprint(training_table_path))
sn_targets = load_csv(sn_targets_path, file_fingerprint(sn_targets_path))
sn_points = load_csv(sn_points_path, file_fingerprint(sn_points_path))
sn_review_queue = load_csv(sn_review_queue_path, file_fingerprint(sn_review_queue_path))
sn_audit_summary = load_json(sn_audit_summary_path, file_fingerprint(sn_audit_summary_path))
sn_model_summary = load_csv(sn_model_summary_path, file_fingerprint(sn_model_summary_path))
sn_model_prediction_grid = load_csv(sn_model_prediction_grid_path, file_fingerprint(sn_model_prediction_grid_path))
sn_model_artifact = load_json(sn_model_artifact_path, file_fingerprint(sn_model_artifact_path))
supporting_literature = build_supporting_literature_table()

ACADEMIC_COLORS = ["#2f5d62", "#5b7f95", "#8a6f3d", "#6b7280", "#a44a3f", "#7d8f69"]
FEASIBILITY_COLORS = {
    "feasible under selected constraints": "#2f5d62",
    "conditional under selected constraints": "#8a6f3d",
    "limited by selected furnace range": "#a44a3f",
}
THERMAL_STEP_COLORS = {
    "Ramp to HIP": "#3b5b8a",
    "HIP hold": "#7f3b08",
    "Ramp to homogenisation": "#5f6f52",
    "Transition to homogenisation": "#5f6f52",
    "Homogenisation hold": "#8a6f3d",
    "Ramp to solution treatment": "#2f5d62",
    "Transition to solution treatment": "#2f5d62",
    "Solution treatment hold": "#1f7a63",
    "Transition to first ageing": "#5b7f95",
    "First ageing hold": "#a6761d",
    "Transition to second ageing": "#6b7280",
    "Second ageing hold": "#a44a3f",
    "Transition to ageing": "#5b7f95",
    "Ageing hold": "#a6761d",
    "Final cooling": "#374151",
}
FATIGUE_SCHEDULE_COLUMN_CONFIG = {
    "stress_amplitude_MPa": st.column_config.NumberColumn("Stress amplitude, sigma_a (MPa)"),
    "stress_ratio_R": st.column_config.NumberColumn("Stress ratio, R"),
    "sigma_max_MPa": st.column_config.NumberColumn("Maximum stress, sigma_max (MPa)"),
    "sigma_min_MPa": st.column_config.NumberColumn("Minimum stress, sigma_min (MPa)"),
    "sigma_mean_MPa": st.column_config.NumberColumn("Mean stress, sigma_mean (MPa)"),
    "target_runout_cycles": st.column_config.TextColumn("Target runout cycles"),
    "interpretation": st.column_config.TextColumn("Interpretation"),
}


def _joined_available(values: pd.Series) -> str:
    available = sorted(str(value) for value in values.dropna().unique())
    return ", ".join(available) if available else "not available"


def _recommendation_status(
    exact_match: bool,
    out_of_grid_fields: list[dict[str, str]],
    fallback_scope: str,
) -> dict[str, object]:
    if exact_match:
        note = "The selected input combination is represented in the reviewed recommendation grid."
    else:
        note = (
            "The selected input combination is outside the reviewed recommendation grid. "
            "Recommendations below use the closest available evidence subset and should be treated as extrapolative screening guidance."
        )
    return {
        "exact_match": exact_match,
        "fallback_scope": fallback_scope,
        "selection_note": note,
        "out_of_grid_fields": out_of_grid_fields,
    }


def _out_of_grid_fields(
    recommendations: pd.DataFrame,
    target: str,
    allow_hip: bool,
    confidence_mode: str,
) -> list[dict[str, str]]:
    fields: list[dict[str, str]] = []
    if recommendations.empty:
        return fields

    available_targets = recommendations["target"].dropna().astype(str)
    if target not in set(available_targets):
        fields.append(
            {
                "field": "Primary design objective",
                "selected": str(target),
                "available": _joined_available(available_targets),
                "interpretation": "No direct route ranking exists for this objective; the closest available objective is used for screening.",
            }
        )

    target_rows = recommendations[recommendations["target"] == target]
    hip_scope = target_rows if not target_rows.empty else recommendations
    available_hip = hip_scope["allow_hip"].dropna().astype(bool)
    if bool(allow_hip) not in set(available_hip):
        fields.append(
            {
                "field": "HIP benchmark inclusion",
                "selected": str(bool(allow_hip)),
                "available": _joined_available(available_hip.astype(str)),
                "interpretation": "The requested HIP setting is not represented for the selected objective; available routes are retained with explicit feasibility notes.",
            }
        )

    mode_scope = recommendations[
        (recommendations["target"] == target) & (recommendations["allow_hip"].astype(bool) == bool(allow_hip))
    ]
    if mode_scope.empty:
        mode_scope = target_rows if not target_rows.empty else recommendations
    available_modes = mode_scope["confidence_mode"].dropna().astype(str)
    if confidence_mode not in set(available_modes):
        fields.append(
            {
                "field": "Decision posture",
                "selected": str(confidence_mode),
                "available": _joined_available(available_modes),
                "interpretation": "The requested decision posture is not available for this evidence subset; the nearest reviewed posture is used.",
            }
        )
    return fields


def _select_recommendation_subset(
    recommendations: pd.DataFrame,
    target: str,
    allow_hip: bool,
    confidence_mode: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if recommendations.empty:
        return recommendations.copy(), _recommendation_status(False, [], "empty evidence base")

    exact = recommendations[
        (recommendations["target"] == target)
        & (recommendations["allow_hip"].astype(bool) == bool(allow_hip))
        & (recommendations["confidence_mode"] == confidence_mode)
    ].copy()
    if not exact.empty:
        return exact, _recommendation_status(True, [], "exact selection")

    out_of_grid = _out_of_grid_fields(recommendations, target, allow_hip, confidence_mode)
    fallback_rules = [
        (
            "same objective and HIP setting with nearest available decision posture",
            (recommendations["target"] == target) & (recommendations["allow_hip"].astype(bool) == bool(allow_hip)),
        ),
        (
            "same objective with available HIP setting",
            recommendations["target"] == target,
        ),
        (
            "balanced objective with selected HIP setting",
            (recommendations["target"] == "balanced") & (recommendations["allow_hip"].astype(bool) == bool(allow_hip)),
        ),
        (
            "balanced non-HIP evidence subset",
            (recommendations["target"] == "balanced") & (recommendations["allow_hip"].astype(bool) == False),
        ),
    ]
    for scope, mask in fallback_rules:
        subset = recommendations[mask].copy()
        if not subset.empty:
            return subset, _recommendation_status(False, out_of_grid, scope)
    return recommendations.copy(), _recommendation_status(False, out_of_grid, "complete evidence base")


def _build_example_input_combinations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "Local non-HIP baseline validation",
                "example_inputs": "Primary objective: balanced; HIP benchmark: off; furnace: up to 1065 C; surface: machined; orientation: vertical",
                "expected_route_family": "ST_DA or CUSTOM_ST_DA",
                "interpretation": "Suitable for a practical first validation set because it stays within common non-HIP furnace capability.",
            },
            {
                "scenario": "Lower-temperature screening route",
                "example_inputs": "Primary objective: strength; HIP benchmark: off; furnace: up to 980 C; surface: machined; cycle limit: 20 h",
                "expected_route_family": "DA or lower-temperature ST_DA",
                "interpretation": "Useful when high-temperature homogenisation is not available; interpret fatigue response cautiously.",
            },
            {
                "scenario": "Segregation-control comparison",
                "example_inputs": "Primary objective: ductility; HIP benchmark: off; furnace: up to 1100 C; section size: moderate section",
                "expected_route_family": "HA_ST_DA or ST_DA",
                "interpretation": "Useful when the experiment is designed to test Laves/Nb-rich segregation control against a simpler route.",
            },
            {
                "scenario": "Literature benchmark with HIP",
                "example_inputs": "Primary objective: balanced; HIP benchmark: on; furnace: not specified; surface: machined",
                "expected_route_family": "HIP_ST_DA, HIP_DA, and non-HIP comparators",
                "interpretation": "Use only as a comparison case when local HIP processing is unavailable.",
            },
        ]
    )


if build_example_input_combinations is None:
    build_example_input_combinations = _build_example_input_combinations
if select_recommendation_subset is None:
    select_recommendation_subset = _select_recommendation_subset


def _build_notation_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"term": "LPBF", "meaning": "laser powder bed fusion", "context": "Additive-manufacturing process used for the Inconel 718 specimens."},
            {"term": "SLM", "meaning": "selective laser melting", "context": "Earlier or alternative literature term often used for LPBF."},
            {"term": "IN718", "meaning": "Inconel 718", "context": "Nickel-base superalloy considered in the recommendation framework."},
            {"term": "HIP", "meaning": "hot isostatic pressing", "context": "Porosity-reduction route retained as a benchmark, not as the local default."},
            {"term": "ST", "meaning": "solution treatment", "context": "High-temperature heat-treatment step used before ageing in several route labels."},
            {"term": "DA", "meaning": "double ageing", "context": "Ageing sequence used to promote gamma-prime and gamma-double-prime strengthening."},
            {"term": "HA", "meaning": "homogenisation anneal", "context": "Higher-temperature step used to reduce segregation before solution treatment and ageing."},
            {"term": "UTS", "meaning": "ultimate tensile strength", "context": "Static tensile indicator reported in MPa."},
            {"term": "YS", "meaning": "yield strength", "context": "Static tensile indicator reported in MPa."},
            {"term": "S-N", "meaning": "stress-life fatigue relationship", "context": "Fatigue representation relating cyclic stress to cycles to failure."},
            {"term": "SEM", "meaning": "scanning electron microscopy", "context": "Microstructural and fracture-surface characterisation method."},
            {"term": "EDS", "meaning": "energy-dispersive X-ray spectroscopy", "context": "Composition-sensitive method used with SEM to assess segregation."},
            {"term": "MPa", "meaning": "megapascal", "context": "Stress unit used for tensile and fatigue quantities."},
            {"term": "wt.%", "meaning": "weight percent", "context": "Composition unit used for nominal chemistry inputs."},
        ]
    )


def _build_equation_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "Larson-Miller thermal dose",
                "latex": r"P = T_K \left(C + \log_{10}(t_h)\right)",
                "definition": "Variables are defined below; this term summarises temperature-time exposure for heat-treatment features.",
            },
            {
                "name": "Arrhenius thermal activation index",
                "latex": r"A = \sum_i t_i \exp\left[-\frac{Q}{R}\left(\frac{1}{T_i} - \frac{1}{T_{ref}}\right)\right]",
                "definition": "Variables are defined below; this index represents thermally activated exposure relative to a reference temperature.",
            },
            {
                "name": "Basquin fatigue relation",
                "latex": r"\sigma_a = \sigma_f' \left(2N_f\right)^b",
                "definition": "Variables are defined below; current S-N fitting is limited to reviewed source-specific literature marker points.",
            },
            {
                "name": "Recommendation-index expression",
                "latex": r"s_{rec} = 0.7s_{evidence} + 0.3s_{property}",
                "definition": "Variables are defined below; this combines the evidence-weighted route score with the calibrated static-property index.",
            },
        ]
    )


if build_notation_table is None:
    build_notation_table = _build_notation_table
if build_equation_table is None:
    build_equation_table = _build_equation_table

INPUT_HELP = {
    "target": (
        "Select the property objective used to weight the route ranking. "
        "Balanced combines strength, ductility, evidence support, and feasibility. "
        "Fatigue keeps fatigue-risk modifiers visible but does not fit S-N life. "
        "Strength gives more weight to calibrated UTS and yield-strength estimates. "
        "Ductility gives more weight to calibrated elongation estimates."
    ),
    "initial_state": (
        "Initial condition of the EOS LPBF Inconel 718 specimens before the proposed heat treatment. "
        "As-built indicates no prior thermal relief. Stress relieved indicates residual-stress reduction before the selected route. "
        "Machined indicates surface removal that can reduce roughness-driven fatigue scatter."
    ),
    "section_size": (
        "Representative section size affects thermal-gradient risk and cycle-time interpretation. "
        "Thin coupon means a low-thermal-mass specimen, approximately up to 3 mm wall thickness or up to 5 mm gauge diameter. "
        "Thin sections are less likely to require extended soak allowances. Moderate sections represent typical test coupons. "
        "Large thermal mass flags possible temperature-lag and cooling-uniformity concerns. Use not specified when dimensions are not yet fixed."
    ),
    "surface_condition": (
        "Surface condition is retained because fatigue response is sensitive to roughness and near-surface defects. "
        "Machined and polished states reduce surface-driven scatter relative to as-built surfaces. "
        "Use not specified only when the final test surface has not been decided."
    ),
    "allow_hip": (
        "Use this switch only when HIP is being considered as a literature benchmark. "
        "Leave it off for the planned local route because HIP is not available locally. "
        "Turning it on allows HIP routes to appear as comparison cases, not as primary recommendations."
    ),
    "mode": (
        "Decision posture controls how strongly the framework penalises limited evidence and local constraints. "
        "Conservative favours better-supported routes. Balanced is the default compromise. "
        "Exploratory allows narrower-evidence routes when they may be scientifically informative."
    ),
    "furnace_limit": (
        "Available furnace range is used to penalise routes that exceed local equipment capability. "
        "Up to 980 C favours lower-temperature solution or ageing routes. Up to 1065 C allows standard non-HIP solution treatment. "
        "Up to 1100 C allows higher homogenisation-style schedules. Use not specified only for early planning."
    ),
    "build_orientation": (
        "Build orientation is retained as a fatigue-risk modifier. "
        "Coordinate convention: X and Y lie in the build plate; Z is the build direction. "
        "Vertical, horizontal, and mixed orientations can show different defect alignment, surface exposure, and fatigue scatter. "
        "The current model records the orientation for risk interpretation rather than fitting orientation-specific fatigue life."
    ),
    "furnace_limit_C": (
        "Maximum furnace temperature available for practical validation. "
        "The framework compares this value with each route's maximum treatment temperature and applies feasibility penalties when the route exceeds local capability."
    ),
    "maximum_cycle_hours": (
        "Maximum total hold time that can be scheduled for a single heat-treatment route. "
        "This is used to penalise long homogenisation or HIP-style schedules and to estimate practical furnace occupancy."
    ),
    "target_life_cycles": (
        "Target fatigue life is used only as a validation objective. "
        "The current model does not infer fatigue life from tensile properties; fatigue claims require local S-N testing and defect characterisation."
    ),
    "stress_ratio_R": (
        "Stress ratio R is the minimum cyclic stress divided by the maximum cyclic stress. "
        "Use R = 0.1 when matching the common tension-tension fatigue condition in the current LPBF Inconel 718 evidence table. "
        "Use a different value only when the planned validation test uses a different loading ratio."
    ),
    "cooling_condition": (
        "Cooling condition affects residual stress relief, precipitation response, and practical repeatability. "
        "Controlled furnace cooling is conservative and repeatable. Air cooling is practical for many non-HIP routes. "
        "Water quenching may alter residual stress and distortion risk and should be recorded explicitly."
    ),
    "chemistry_record": (
        "Enable this only when composition values will be recorded for the experimental batch. "
        "These values are retained for traceability and later model expansion; they do not currently override the calibrated property estimates."
    ),
    "niobium": (
        "Nb + Ta content is relevant to Laves/Nb-rich segregation and gamma-double-prime strengthening. "
        "Record the nominal powder or certificate value when available."
    ),
    "aluminium": (
        "Al contributes to gamma-prime precipitation. "
        "Record it with Ti and Nb so later analysis can separate composition effects from heat-treatment effects."
    ),
    "titanium": (
        "Ti contributes to gamma-prime precipitation and should be recorded with Al and Nb when available. "
        "The current route ranking stores this value for traceability and future calibration."
    ),
}


def academic_layout(fig: go.Figure, title: str, height: int | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        title_font=dict(size=17, color="#1f2933"),
        paper_bgcolor="#f8f8f5",
        plot_bgcolor="#f8f8f5",
        font=dict(family="Arial", color="#1f2933", size=13),
        margin=dict(l=20, r=20, t=55, b=35),
        legend_title_text="",
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig


def build_manual_context_from_inputs(**values) -> ManualInputContext:
    supported = set(inspect.signature(ManualInputContext).parameters)
    return ManualInputContext(**{key: value for key, value in values.items() if key in supported})


with st.sidebar:
    st.header("Configuration")
    st.caption("Persistent input parameters for all recommendation, process-window, and property views.")
    target = st.selectbox(
        "Primary design objective",
        ["balanced", "fatigue", "strength", "ductility"],
        help=INPUT_HELP["target"],
    )
    allow_hip = st.toggle(
        "Include HIP benchmark routes",
        value=False,
        help=INPUT_HELP["allow_hip"],
    )
    mode = st.selectbox(
        "Decision posture",
        ["conservative", "balanced", "exploratory"],
        index=1,
        help=INPUT_HELP["mode"],
    )
    furnace_limit = st.selectbox(
        "Available furnace range",
        ["up to 980 C", "up to 1065 C", "up to 1100 C", "not specified"],
        index=3,
        help=INPUT_HELP["furnace_limit"],
    )
    build_orientation = st.selectbox(
        "Build orientation",
        ["vertical", "horizontal", "mixed", "not specified"],
        help=INPUT_HELP["build_orientation"],
    )
    initial_state = st.selectbox(
        "Initial material state",
        ["EOS-like LPBF, as-built", "EOS-like LPBF, stress relieved", "EOS-like LPBF, machined"],
        help=INPUT_HELP["initial_state"],
    )
    section_size = st.selectbox(
        "Representative section size",
        ["thin section", "moderate section", "large thermal mass", "not specified"],
        index=3,
        help=INPUT_HELP["section_size"],
    )
    surface_condition = st.selectbox(
        "Surface condition",
        ["machined", "polished", "as-built", "not specified"],
        help=INPUT_HELP["surface_condition"],
    )

    st.markdown("#### Manual experimental inputs")
    furnace_defaults = {"up to 980 C": 980, "up to 1065 C": 1065, "up to 1100 C": 1100, "not specified": 1100}
    furnace_limit_C = st.number_input(
        "Maximum furnace temperature (C)",
        min_value=600,
        max_value=1250,
        value=furnace_defaults[furnace_limit],
        step=5,
        help=INPUT_HELP["furnace_limit_C"],
    )
    maximum_cycle_hours = st.number_input(
        "Maximum practical cycle time (h)",
        min_value=1.0,
        max_value=96.0,
        value=20.0,
        step=1.0,
        help=INPUT_HELP["maximum_cycle_hours"],
    )
    target_life_cycles = st.number_input(
        "Target fatigue life, if applicable (cycles)",
        min_value=0,
        max_value=100000000,
        value=1000000,
        step=100000,
        help=INPUT_HELP["target_life_cycles"],
    )
    stress_ratio_R = st.number_input(
        "Fatigue stress ratio, R",
        min_value=-1.0,
        max_value=0.9,
        value=0.1,
        step=0.1,
        format="%.2f",
        help=INPUT_HELP["stress_ratio_R"],
    )
    cooling_condition = st.selectbox(
        "Cooling condition available",
        ["controlled furnace cooling", "air cooling", "water quench", "not specified"],
        help=INPUT_HELP["cooling_condition"],
    )
    chemistry_record = st.toggle(
        "Record nominal chemistry inputs",
        value=False,
        help=INPUT_HELP["chemistry_record"],
    )
    niobium = aluminium = titanium = None
    if chemistry_record:
        niobium = st.number_input(
            "Nb + Ta (wt.%)",
            min_value=0.0,
            max_value=8.0,
            value=5.1,
            step=0.1,
            help=INPUT_HELP["niobium"],
        )
        aluminium = st.number_input(
            "Al (wt.%)",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.05,
            help=INPUT_HELP["aluminium"],
        )
        titanium = st.number_input(
            "Ti (wt.%)",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.05,
            help=INPUT_HELP["titanium"],
        )

    with st.expander("Current input context", expanded=False):
        st.write(
            {
                "primary_design_objective": target,
                "hip_benchmark_included": allow_hip,
                "decision_posture": mode,
                "available_furnace_range": furnace_limit,
                "maximum_furnace_temperature_C": furnace_limit_C,
                "maximum_practical_cycle_time_h": maximum_cycle_hours,
                "initial_material_state": initial_state,
                "representative_section_size": section_size,
                "surface_condition": surface_condition,
                "build_orientation": build_orientation,
                "cooling_condition_available": cooling_condition,
                "target_fatigue_life_cycles": target_life_cycles,
                "fatigue_stress_ratio_R": stress_ratio_R,
            }
        )
    with st.expander("Show example combinations", expanded=False):
        st.markdown("#### Example input combinations")
        st.dataframe(build_example_input_combinations(), width="stretch")
    with st.expander("Available treatment routes", expanded=False):
        st.markdown("#### Available treatment routes in the reviewed evidence base")
        if recs.empty:
            st.info("The reviewed recommendation table is not available in this session.")
        else:
            route_table_columns = [
                "target",
                "allow_hip",
                "confidence_mode",
                "ht_class",
                "selected_recipe_summary",
                "recommended_peak_temperature_C",
                "recommended_total_hold_h",
                "temperature_time_window",
                "confidence",
            ]
            st.dataframe(
                recs.reindex(columns=route_table_columns).drop_duplicates(),
                width="stretch",
            )

manual_context = build_manual_context_from_inputs(
    furnace_limit_C=int(furnace_limit_C),
    maximum_cycle_hours=float(maximum_cycle_hours),
    section_size=section_size,
    surface_condition=surface_condition,
    build_orientation=build_orientation,
    initial_material_state=initial_state,
    cooling_condition=cooling_condition,
    target_life_cycles=int(target_life_cycles) if target_life_cycles else None,
    stress_ratio_R=float(stress_ratio_R),
    niobium_wt_percent=niobium,
    aluminium_wt_percent=aluminium,
    titanium_wt_percent=titanium,
)
fatigue_schedule = build_fatigue_validation_schedule(
    stress_ratio_R=float(stress_ratio_R),
    target_life_cycles=int(target_life_cycles) if target_life_cycles else None,
)
sn_training_status = build_sn_training_status(sn_points=sn_points, sn_targets=sn_targets)
sn_model_available = not sn_model_summary.empty and not sn_model_prediction_grid.empty
input_conditions = {
    "Primary design objective": target,
    "Initial material state": initial_state,
    "Representative section size": section_size,
    "HIP benchmark inclusion": allow_hip,
    "Decision posture": mode,
    "Available furnace range": furnace_limit,
    "Maximum furnace temperature": f"{int(furnace_limit_C)} C",
    "Maximum practical cycle time": f"{float(maximum_cycle_hours):.1f} h",
    "Surface condition": surface_condition,
    "Build orientation": build_orientation,
    "Cooling condition": cooling_condition,
    "Target fatigue life": f"{int(target_life_cycles):,} cycles" if target_life_cycles else "not specified",
    "Fatigue stress ratio": f"R = {float(stress_ratio_R):g}",
}
if niobium is not None:
    input_conditions["Nb + Ta"] = f"{float(niobium):.2f} wt.%"
if aluminium is not None:
    input_conditions["Al"] = f"{float(aluminium):.2f} wt.%"
if titanium is not None:
    input_conditions["Ti"] = f"{float(titanium):.2f} wt.%"

filtered = pd.DataFrame()
adjusted = pd.DataFrame()
selection_status: dict[str, object] = {
    "exact_match": True,
    "selection_note": (
        "The selected input combination is outside the reviewed recommendation grid. "
        "Recommendations below use the closest available evidence subset and should be treated as extrapolative screening guidance."
    ),
    "out_of_grid_fields": [],
    "fallback_scope": "not evaluated",
}
if not recs.empty:
    filtered, selection_status = select_recommendation_subset(recs, target, allow_hip, mode)
    adjusted = apply_manual_inputs(filtered, manual_context)
    adjusted = apply_ml_property_ranking(adjusted, route_predictions, target)

top_row = adjusted.sort_values("ml_assisted_rank").iloc[0] if not adjusted.empty else None

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Decision Dossier",
    "Evidence Base",
    "Process Window",
    "Property Assessment",
    "Validation Plan",
    "Help and Scientific Basis",
])

with tab1:
    st.subheader("Decision Support Dossier")
    if top_row is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended route", str(top_row["ht_class"]), help="The highest-ranked heat treatment route based on the selected objective and constraints.")
        m2.metric("Recommendation index", f"{float(top_row['ml_assisted_score']):.2f}", help="Composite score balancing property estimates, evidence confidence, and local feasibility.")
        m3.metric("Evidence confidence", str(top_row["confidence"]), help="Categorical confidence derived from the number of supporting literature records.")
        occupancy = top_row.get("estimated_furnace_occupancy_h", "not assessed")
        m4.metric("Estimated furnace occupancy", f"{float(occupancy):.1f} h" if pd.notna(occupancy) and occupancy != "not assessed" else "not assessed", help="Total expected hours of furnace time, excluding ramp rates.")
        st.markdown("#### Proposed experimental recipe")
        recipe_cols = st.columns(3)
        recipe_cols[0].metric(
            "Peak temperature",
            f"{int(top_row['recommended_peak_temperature_C'])} C" if pd.notna(top_row.get("recommended_peak_temperature_C")) else "not specified",
            help="Concrete maximum temperature used for local feasibility ranking.",
        )
        recipe_cols[1].metric(
            "Total hold time",
            f"{float(top_row['recommended_total_hold_h']):.1f} h" if pd.notna(top_row.get("recommended_total_hold_h")) else "not specified",
            help="Total scheduled hold time for the proposed experimental recipe.",
        )
        recipe_cols[2].metric(
            "Fatigue validation context",
            f"R = {stress_ratio_R:g}; Nf = {int(target_life_cycles):,}",
            help="Stress ratio and target cycles for validation planning. This is not a fatigue-life prediction.",
        )
        st.write(str(top_row.get("selected_recipe_summary", "No proposed recipe is available.")))
        st.markdown("#### Fatigue validation stress schedule")
        st.caption(
            "These stress levels are proposed for experimental validation at the selected stress ratio. "
            "The table does not report predicted fatigue life."
        )
        st.dataframe(fatigue_schedule, width="stretch", column_config=FATIGUE_SCHEDULE_COLUMN_CONFIG)
        st.markdown("#### Text recommendation")
        st.info(generate_text_recommendation(top_row, manual_context))
        st.caption(
            "This text is a traceable recommendation summary. It should be read with the ranked route table, evidence base, and validation plan."
        )
        st.divider()
        st.markdown("#### Printable recommendation report")
        if sn_model_available:
            st.caption(
                "The report below is formatted for research-team discussion and local printing. "
                "S-N curves are trained only for source-specific literature conditions and are not design allowables."
            )
        else:
            st.caption(
                "The report below is formatted for research-team discussion and local printing. "
                "S-N curves have not yet been trained; Fatigue life is not predicted in the current release."
            )
        report_markdown = build_printable_report_safely(
            input_conditions=input_conditions,
            top_row=top_row,
            context=manual_context,
            fatigue_schedule=fatigue_schedule,
            sn_status=sn_training_status,
            experiments=build_must_have_experiments(str(top_row["ht_class"]), allow_hip),
        )
        technician_sheet_markdown = extract_technician_sheet(report_markdown)
        st.markdown("#### Print / export")
        print_col, full_report_col, technician_col = st.columns(3)
        with print_col:
            st.markdown(
                """
                <button onclick="window.print()" style="
                    width: 100%;
                    padding: 0.58rem 0.75rem;
                    border: 1px solid #2f5d62;
                    border-radius: 6px;
                    background: #2f5d62;
                    color: white;
                    font: 600 14px Arial, sans-serif;
                    cursor: pointer;">
                    Print visible report
                </button>
                """,
                unsafe_allow_html=True,
            )
        with full_report_col:
            st.download_button(
                "Download full report",
                report_markdown.encode("utf-8"),
                file_name=f"ht_advisor_{str(top_row['ht_class']).lower()}_recommendation_report.md",
                mime="text/markdown",
                help="Download a Markdown report containing input conditions, recommendation details, technician instructions, static estimates, validation schedule, and S-N model status.",
                width="stretch",
            )
        with technician_col:
            st.download_button(
                "Download technician sheet",
                technician_sheet_markdown.encode("utf-8"),
                file_name=f"ht_advisor_{str(top_row['ht_class']).lower()}_technician_sheet.md",
                mime="text/markdown",
                help="Download only the heat-treatment instruction sheet for technician review and completion.",
                width="stretch",
            )
        st.caption(
            "Use the print button for the visible report, or download the Markdown files for review, completion, and local document control."
        )
        with st.container(border=True):
            st.markdown("<div class='print-report'>", unsafe_allow_html=True)
            st.markdown("### Process & Material Specifications")
            st.markdown("#### Full input conditions")
            st.dataframe(pd.DataFrame([{"condition": key, "value": str(value)} for key, value in input_conditions.items()]), width="stretch", hide_index=True)
            st.markdown("#### Recommended heat-treatment route")
            report_cols = st.columns(4)
            report_cols[0].metric("Route", str(top_row["ht_class"]))
            report_cols[1].metric(
                "Peak temperature",
                f"{int(top_row['recommended_peak_temperature_C'])} C" if pd.notna(top_row.get("recommended_peak_temperature_C")) else "not specified",
            )
            report_cols[2].metric(
                "Total hold time",
                f"{float(top_row['recommended_total_hold_h']):.1f} h" if pd.notna(top_row.get("recommended_total_hold_h")) else "not specified",
            )
            report_cols[3].metric("Recommendation index", f"{float(top_row['ml_assisted_score']):.2f}")
            st.write(str(top_row.get("selected_recipe_summary", top_row.get("temperature_time_window", "No proposed recipe is available."))))
            report_cycle_rows = build_thermal_cycle_rows(
                str(top_row["ht_class"]),
                str(top_row.get("selected_recipe_summary", top_row.get("temperature_time_window", ""))),
            )
            report_cycle_segment_rows = build_thermal_cycle_segment_rows(
                str(top_row["ht_class"]),
                str(top_row.get("selected_recipe_summary", top_row.get("temperature_time_window", ""))),
            )

            st.markdown("#### Technician heat-treatment instruction sheet")
            st.write(
                "Instruction status: draft work instruction for technician review; verify against the local furnace standard operating procedure before processing. "
                "Do not begin heat treatment until the required blanks below are completed and the route is approved by the process owner."
            )
            tech_cols = st.columns(2)
            with tech_cols[0]:
                st.markdown("##### Material and specimen identification")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"field": "Specimen or batch ID", "value": "to be completed"},
                            {"field": "Alloy and process", "value": "LPBF Inconel 718"},
                            {"field": "Initial material state", "value": str(initial_state)},
                            {"field": "Build orientation", "value": str(build_orientation)},
                            {"field": "Surface condition", "value": str(surface_condition)},
                            {"field": "Representative section size", "value": str(section_size)},
                            {"field": "Number of specimens", "value": "to be completed"},
                            {"field": "Drawing or coupon geometry reference", "value": "to be completed"},
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            with tech_cols[1]:
                st.markdown("##### Equipment and furnace programme")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"field": "Furnace ID", "value": "to be completed"},
                            {"field": "Furnace programme ID", "value": "to be completed"},
                            {"field": "Maximum permitted furnace temperature", "value": f"{int(furnace_limit_C)} C"},
                            {"field": "Maximum practical cycle time", "value": f"{float(maximum_cycle_hours):.1f} h"},
                            {"field": "Planning ramp rate", "value": "10 C/min for dashboard occupancy estimate; record actual ramp"},
                            {"field": "Furnace atmosphere, vacuum, or shielding gas", "value": "to be completed"},
                            {"field": "Thermocouple or witness coupon location", "value": "to be completed"},
                            {"field": "Final cooling method", "value": str(cooling_condition)},
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            st.markdown("##### Nominal thermal programme")
            if report_cycle_rows.empty:
                st.info("Nominal thermal programme could not be parsed automatically; complete the furnace programme manually before processing.")
            else:
                cycle_reset = report_cycle_rows.reset_index(drop=True)
                hold_steps = []
                for idx, item in cycle_reset[cycle_reset["stage"].astype(str).str.startswith("hold")].iterrows():
                    previous_elapsed = float(cycle_reset.iloc[idx - 1]["elapsed_h"]) if idx > 0 else 0.0
                    hold_steps.append(
                        {
                            "Step": len(hold_steps) + 1,
                            "Action": "Hold at temperature",
                            "Set point": f"{int(item['temperature_C'])} C",
                            "Hold time": f"{float(item['elapsed_h']) - previous_elapsed:g} h",
                            "Cooling or transfer note": "Proceed directly to the next specified step unless the local furnace procedure requires an intermediate cool.",
                        }
                    )
                st.dataframe(pd.DataFrame(hold_steps), width="stretch", hide_index=True)
            st.markdown("##### Stage interpretation through the profile")
            if str(top_row["ht_class"]) == "ST_DA":
                st.write(
                    "The full route is ST_DA: solution treatment is completed first, followed by the two ageing holds that make up double ageing."
                )
                st.markdown(
                    """
                    | Profile region | When it occurs | Purpose in this route |
                    | --- | --- | --- |
                    | Solution treatment | Ramp to 980 C, then hold 980 C for 1 h | Sets the solution-treated condition before ageing; intended to reduce as-built heat-treatment sensitivity before precipitation ageing. |
                    | Double ageing - first hold | After solution treatment, hold 720 C for 8 h | First ageing hold for precipitation strengthening. |
                    | Double ageing - second hold | Then hold 620 C for 8 h | Second ageing hold completing the double-ageing sequence. |
                    | Final cooling | After the 620 C hold | Use controlled furnace cooling unless the process owner approves a different cooling condition. |
                    """
                )
            else:
                st.write("Stage interpretation should be reviewed against the selected route and local furnace procedure before processing.")
            st.markdown("##### Required process records")
            st.markdown(
                """
                - Confirm furnace calibration is in date before loading.
                - Record actual ramp rate, soak start time, soak end time, and actual cooling condition.
                - Record specimen placement, fixture or tray arrangement, and thermocouple or witness coupon position.
                - Record any deviation from the programme before using the specimens for property claims.
                - Operator sign-off: to be completed.
                """
            )

            plot_col_1, plot_col_2 = st.columns(2)
            with plot_col_1:
                st.markdown("#### Heat-treatment profile plot")
                if report_cycle_segment_rows.empty:
                    st.info("Thermal-cycle profile is unavailable for the selected route.")
                else:
                    report_cycle_fig = px.line(
                        report_cycle_segment_rows,
                        x="elapsed_h",
                        y="temperature_C",
                        markers=True,
                        color="segment_label",
                        line_group="segment_id",
                        color_discrete_map=THERMAL_STEP_COLORS,
                        labels={"elapsed_h": "Time, t (h)", "temperature_C": "Temperature, T (C)", "segment_label": "Thermal step"},
                    )
                    report_cycle_fig.update_traces(
                        hovertemplate="Thermal step: %{fullData.name}<br>Time: %{x:.2f} h<br>Temperature: %{y:.0f} C<extra></extra>"
                    )
                    st.plotly_chart(academic_layout(report_cycle_fig, "Nominal recommended thermal cycle", height=390), width="stretch")
                    st.caption("Nominal profile shown; actual thermal history depends on furnace thermal mass, part geometry, and thermocouple placement.")
            with plot_col_2:
                st.markdown("#### Fatigue validation schedule plot")
                schedule_long = fatigue_schedule.melt(
                    id_vars=["stress_amplitude_MPa", "target_runout_cycles"],
                    value_vars=["sigma_max_MPa", "sigma_mean_MPa", "sigma_min_MPa"],
                    var_name="stress_quantity",
                    value_name="stress_MPa",
                )
                schedule_fig = px.line(
                    schedule_long,
                    x="stress_amplitude_MPa",
                    y="stress_MPa",
                    color="stress_quantity",
                    markers=True,
                    color_discrete_sequence=ACADEMIC_COLORS,
                    labels={
                        "stress_amplitude_MPa": "Stress amplitude, sigma_a (MPa)",
                        "stress_MPa": "Applied stress, sigma (MPa)",
                        "stress_quantity": "Stress quantity",
                    },
                )
                schedule_fig.update_traces(
                    hovertemplate="Stress amplitude: %{x:.0f} MPa<br>Applied stress: %{y:.0f} MPa<br>Quantity: %{fullData.name}<extra></extra>"
                )
                st.plotly_chart(academic_layout(schedule_fig, "Validation stress schedule", height=390), width="stretch")
                st.caption(
                    f"Stress levels are test conditions for R = {float(stress_ratio_R):g}; they do not indicate survival to "
                    f"{int(target_life_cycles):,} cycles."
                )

            st.markdown("#### Expected static-property estimates")
            property_rows = []
            property_specs = [
                ("UTS", "predicted_UTS_MPa", "MPa"),
                ("YS", "predicted_YS_MPa", "MPa"),
                ("Elongation", "predicted_elongation_pct", "%"),
            ]
            for label, column, unit in property_specs:
                if column in top_row and pd.notna(top_row.get(column)):
                    property_rows.append(
                        {
                            "property": label,
                            "estimate": float(top_row[column]),
                            "lower": float(top_row.get(f"{column}_lower", top_row[column])),
                            "upper": float(top_row.get(f"{column}_upper", top_row[column])),
                            "unit": unit,
                        }
                    )
            if property_rows:
                property_df = pd.DataFrame(property_rows)
                property_fig = go.Figure()
                for _, property_row in property_df.iterrows():
                    property_fig.add_trace(
                        go.Bar(
                            x=[property_row["property"]],
                            y=[property_row["estimate"]],
                            error_y=dict(
                                type="data",
                                array=[max(property_row["upper"] - property_row["estimate"], 0.0)],
                                arrayminus=[max(property_row["estimate"] - property_row["lower"], 0.0)],
                            ),
                            name=f"{property_row['property']} ({property_row['unit']})",
                            hovertemplate=(
                                "Property: %{x}<br>"
                                "Estimate: %{y:.1f}<br>"
                                "<extra></extra>"
                            ),
                        )
                    )
                property_fig.update_yaxes(title="Estimated value")
                st.plotly_chart(academic_layout(property_fig, "Expected static-property estimates with empirical bounds", height=390), width="stretch")
                st.dataframe(property_df, width="stretch", hide_index=True)
            else:
                st.info("No calibrated static-property estimates are available for the selected route.")

            st.markdown("#### S-N training status")
            st.write(str(sn_training_status["status_message"]))
            st.write(str(sn_training_status["report_note"]))
            if sn_model_available:
                st.write(
                    f"Trained S-N module: {len(sn_model_summary)} source-specific physics-constrained Basquin curves "
                    f"from {len(sn_points)} reviewed marker points. Not a design allowable."
                )
            st.markdown("#### Markdown report preview")
            st.markdown(report_markdown)
            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Current input context", expanded=False):
        st.write(
            {
                "primary_design_objective": target,
                "initial_material_state": initial_state,
                "representative_section_size": section_size,
                "hip_benchmark_included": allow_hip,
                "decision_posture": mode,
                "available_furnace_range": furnace_limit,
                "maximum_furnace_temperature_C": furnace_limit_C,
                "maximum_practical_cycle_time_h": maximum_cycle_hours,
                "surface_condition": surface_condition,
                "build_orientation": build_orientation,
                "cooling_condition_available": cooling_condition,
                "target_fatigue_life_cycles": target_life_cycles,
                "fatigue_stress_ratio_R": stress_ratio_R,
            }
        )

    with st.expander("Disclaimer and context", expanded=True):
        if not allow_hip:
            st.info("Local constraint: HIP is not available. Ranked routes therefore prioritise non-HIP heat treatments, while HIP remains available as a benchmark comparison.")
        if trained_model:
            trained_targets = ", ".join(trained_model.get("trained_targets", [])) or "none"
            st.caption(
                f"Calibration status: {trained_model.get('model_status', 'not available')}. "
                f"Calibrated targets: {trained_targets}. This is not a physics-informed neural network."
            )
        st.warning(
            "Scientific interpretation note: recommendations currently optimize static tensile indicators only; "
            "fatigue is defect-controlled in non-HIP LPBF Inconel 718 and must not be inferred from tensile strength alone."
        )
    st.divider()
    st.markdown("#### Ranked treatment routes")
    if filtered.empty:
        st.warning("The reviewed evidence base is empty; treatment-route ranking cannot be computed in this session.")
    else:
        if not bool(selection_status.get("exact_match", True)):
            st.warning(str(selection_status.get("selection_note", "")))
            out_of_grid_fields = selection_status.get("out_of_grid_fields", [])
            if out_of_grid_fields:
                st.markdown("#### Out-of-grid input fields")
                st.dataframe(pd.DataFrame(out_of_grid_fields), width="stretch")
            st.caption(f"Fallback evidence subset: {selection_status.get('fallback_scope', 'closest available evidence subset')}.")

        plot_rows = adjusted.copy()
        if not plot_rows.empty:
            fig = px.bar(
                plot_rows,
                x="ml_assisted_score",
                y="ht_class",
                color="local_feasibility",
                orientation="h",
                text="ml_assisted_score",
                color_discrete_map=FEASIBILITY_COLORS,
                custom_data=["local_feasibility"],
                labels={
                    "ml_assisted_score": "Model-supported recommendation index",
                    "ht_class": "Heat-treatment route",
                    "local_feasibility": "Local feasibility",
                },
            )
            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
                hovertemplate=(
                    "Route: %{y}<br>"
                    "Recommendation index: %{x:.2f}<br>"
                    "Local feasibility: %{customdata[0]}<br>"
                    "<extra></extra>"
                ),
            )
            fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(plot_rows["ht_class"].tolist())))
            st.plotly_chart(academic_layout(fig, "Model-supported heat-treatment ranking for the selected input context", height=520), width="stretch")

            ev_fig = px.scatter(
                plot_rows,
                x="ml_assisted_rank",
                y="evidence_count_seed",
                size="ml_assisted_score",
                color="ht_class",
                color_discrete_sequence=ACADEMIC_COLORS,
                custom_data=["ht_class"],
                labels={"ml_assisted_rank": "Model-supported recommendation rank", "evidence_count_seed": "Supporting records", "ht_class": "Heat-treatment route"},
            )
            ev_fig.update_traces(
                hovertemplate=(
                    "Route: %{customdata[0]}<br>"
                    "Rank: %{x}<br>"
                    "Supporting records: %{y}<br>"
                    "Recommendation index: %{marker.size:.2f}<br>"
                    "<extra></extra>"
                ),
            )
            ev_fig.update_xaxes(dtick=1)
            st.plotly_chart(academic_layout(ev_fig, "Evidence support by recommendation rank", height=420), width="stretch")

            st.divider()
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("#### Recommended-route thermal cycle")
                cycle_source = str(top_row.get("selected_recipe_summary", top_row["temperature_time_window"])) if top_row is not None else ""
                cycle_segment_rows = build_thermal_cycle_segment_rows(str(top_row["ht_class"]), cycle_source) if top_row is not None else pd.DataFrame()
                if cycle_segment_rows.empty:
                    st.info("Thermal-cycle profile is unavailable for the selected route.")
                else:
                    cycle_fig = px.line(
                        cycle_segment_rows,
                        x="elapsed_h",
                        y="temperature_C",
                        markers=True,
                        color="segment_label",
                        line_group="segment_id",
                        color_discrete_map=THERMAL_STEP_COLORS,
                        labels={"elapsed_h": "Elapsed time (h)", "temperature_C": "Temperature (C)", "segment_label": "Thermal step"},
                    )
                    cycle_fig.update_traces(
                        hovertemplate=(
                            "Route: %{customdata[0]}<br>"
                            "Thermal step: %{customdata[1]}<br>"
                            "Elapsed time: %{x:.2f} h<br>"
                            "Temperature: %{y:.0f} C<br>"
                            "<extra></extra>"
                        ),
                        customdata=cycle_segment_rows[["ht_class", "segment_label"]],
                    )
                    st.plotly_chart(academic_layout(cycle_fig, "Time-temperature profile for the top-ranked route", height=420), width="stretch")
            with v2:
                st.markdown("#### Property and evidence trade-off radar")
                radar_rows = build_route_radar_rows(adjusted.sort_values("ml_assisted_rank").head(3))
                if radar_rows.empty:
                    st.info("Radar comparison is unavailable for the selected route set.")
                else:
                    radar_fig = go.Figure()
                    for route, subset in radar_rows.groupby("ht_class"):
                        closed = pd.concat([subset, subset.iloc[[0]]], ignore_index=True)
                        radar_fig.add_trace(
                            go.Scatterpolar(
                                r=closed["value"],
                                theta=closed["axis"],
                                fill="toself",
                                name=str(route),
                                hovertemplate="Route: %{fullData.name}<br>Axis: %{theta}<br>Normalised value: %{r:.1f}<extra></extra>",
                            )
                        )
                    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
                    st.plotly_chart(academic_layout(radar_fig, "Normalised route trade-off comparison", height=420), width="stretch")

            st.markdown("#### Recommendation-index contribution summary")
            contribution_rows = build_recommendation_contribution_rows(top_row) if top_row is not None else pd.DataFrame()
            if not contribution_rows.empty:
                contribution_fig = go.Figure(
                    go.Waterfall(
                        x=contribution_rows["term"],
                        y=contribution_rows["value"],
                        measure=contribution_rows["measure"],
                        connector={"line": {"color": "#6b7280"}},
                        increasing={"marker": {"color": ACADEMIC_COLORS[0]}},
                        totals={"marker": {"color": ACADEMIC_COLORS[2]}},
                        hovertemplate="Term: %{x}<br>Contribution: %{y:.2f}<extra></extra>",
                    )
                )
                st.plotly_chart(academic_layout(contribution_fig, "Evidence and property terms in the selected recommendation index", height=420), width="stretch")

        with st.expander("Generate text recommendation", expanded=False):
            st.markdown("#### Text recommendation")
            st.write(generate_text_recommendation(top_row, manual_context))
            if bool(top_row.get("outside_training_envelope", False)):
                st.warning(str(top_row.get("training_envelope_note", "Extrapolation warning: selected route is outside the reviewed calibration envelope.")))
            st.markdown("#### Must-have experimental validation")
            st.write("These experiments should accompany any result reported from this recommendation.")
            st.write(
                "Required set: as-built baseline; AMS-style standard baseline; framework-recommended route; "
                "hardness and tensile testing; SEM/EDS microstructural assessment; fatigue screening where specimen count and machine time allow."
            )
            st.dataframe(pd.DataFrame(build_must_have_experiments(str(top_row["ht_class"]), allow_hip)), width="stretch")
            st.markdown("#### Stochastic response considerations")
            st.write(
                "The ranking should be interpreted as a candidate selection, not as deterministic performance evidence. "
                "LPBF fatigue and ductility can vary with pore population, surface condition, build orientation, powder history, and local thermal history."
            )
            literature_notes = build_recommendation_literature_notes(target, build_orientation)
            if literature_notes:
                st.markdown("#### Supporting literature for recommendation")
                for note in literature_notes:
                    evidence = supporting_literature[supporting_literature["citation_key"] == note["citation_key"]].iloc[0]
                    st.write(f"**{evidence['display_citation']}** ({evidence['doi']}): {note['note']}")

        with st.expander("Show model specification", expanded=False):
            st.markdown("#### Model specification")
            spec = build_model_specification()
            for key, value in spec.items():
                label = key.replace("_", " ").title()
                if isinstance(value, list):
                    st.markdown(f"**{label}**")
                    for item in value:
                        st.write(f"- {item}")
                else:
                    st.write(f"**{label}:** {value}")
            if trained_model:
                st.markdown("**Calibrated property model**")
                st.write(f"Model family: {trained_model.get('model_family', 'not available')}")
                st.write(f"Calibration rows: {trained_model.get('training_rows_total', 'not available')}")
                st.write(f"Calibrated targets: {', '.join(trained_model.get('trained_targets', [])) or 'none'}")
                st.write("This property model is not a physics-informed neural network; it is an empirically calibrated parametric model.")

        for _, row in adjusted.sort_values("ml_assisted_rank").iterrows():
            with st.container(border=True):
                cols = st.columns([1, 2, 1, 1])
                cols[0].metric(f"Rank {int(row['ml_assisted_rank'])}", row["ht_class"], f"index {row['ml_assisted_score']:.2f}")
                cols[1].write(f"Proposed recipe: {row.get('selected_recipe_summary', row['temperature_time_window'])}")
                cols[2].write(f"Evidence confidence: **{row['confidence']}**")
                cols[3].write(f"Supporting records: **{row['evidence_count_seed']}**")
                st.progress(max(0.0, min(float(row["ml_assisted_score"]), 1.0)), text=f"Recommendation index {float(row['ml_assisted_score']):.2f}")
                st.markdown(f"<span class='route-chip'>{row['local_feasibility']}</span>", unsafe_allow_html=True)
                st.caption(row["recommendation_reason"])
                with st.expander("Rationale and evidence status"):
                    st.write(f"Evidence envelope: **{row['inside_evidence_envelope']}**")
                    st.write(f"Recommended validation recipe: **{row.get('selected_recipe_summary', 'not specified')}**")
                    if pd.notna(row.get("recommended_peak_temperature_C")):
                        st.write(f"Recommended peak temperature: **{int(row['recommended_peak_temperature_C'])} C**")
                    if pd.notna(row.get("recommended_total_hold_h")):
                        st.write(f"Recommended total hold time: **{float(row['recommended_total_hold_h']):.1f} h**")
                    st.write(f"Fatigue validation context: **R = {stress_ratio_R:g}; Nf = {int(target_life_cycles):,} cycles**")
                    st.write(f"Supporting temperature-time window: {row['temperature_time_window']}")
                    if str(row.get("ht_class", "")) == "CUSTOM_ST_DA":
                        st.warning(
                            "Short-cycle solution treatment is an exploratory thin coupon route. "
                            "Use it only for low-thermal-mass coupons, approximately up to 3 mm wall thickness or up to 5 mm gauge diameter, "
                            "and confirm the response by hardness, tensile testing, and microscopy."
                        )
                    st.write(f"Local feasibility: **{row['local_feasibility']}**")
                    st.write(row["constraint_notes"])
                    st.write(f"Property-estimation scope: **{row['ml_assistance_scope']}**")
                    st.write(f"Evidence-weighted score before property-model blend: **{row['adjusted_score']:.2f}**")
                    if "estimated_furnace_occupancy_h" in row and pd.notna(row.get("estimated_furnace_occupancy_h")):
                        st.write(f"Estimated furnace occupancy: **{row['estimated_furnace_occupancy_h']:.2f} h**")
                    if "metallurgical_rule_flags" in row and pd.notna(row.get("metallurgical_rule_flags")):
                        st.write(f"Metallurgical rule flags: {row['metallurgical_rule_flags']}")
                    if bool(row.get("outside_training_envelope", False)):
                        st.warning(str(row.get("training_envelope_note", "Extrapolation warning: this route is outside the reviewed calibration envelope.")))
                    elif "training_envelope_note" in row and pd.notna(row.get("training_envelope_note")):
                        st.write(f"Calibration envelope status: {row['training_envelope_note']}")
                    if "predicted_UTS_MPa" in row and pd.notna(row.get("predicted_UTS_MPa")):
                        st.write(
                            "Empirically calibrated property estimates with Empirical error bounds "
                            f"(bounded to the observed evidence range): UTS {row['predicted_UTS_MPa']:.1f} MPa "
                            f"[{row.get('predicted_UTS_MPa_lower', float('nan')):.1f}, {row.get('predicted_UTS_MPa_upper', float('nan')):.1f}], "
                            f"YS {row.get('predicted_YS_MPa', float('nan')):.1f} MPa "
                            f"[{row.get('predicted_YS_MPa_lower', float('nan')):.1f}, {row.get('predicted_YS_MPa_upper', float('nan')):.1f}], "
                            f"elongation {row.get('predicted_elongation_pct', float('nan')):.1f}% "
                            f"[{row.get('predicted_elongation_pct_lower', float('nan')):.1f}, {row.get('predicted_elongation_pct_upper', float('nan')):.1f}]."
                        )
                        if "empirical_error_bound_note" in row and pd.notna(row.get("empirical_error_bound_note")):
                            st.caption(str(row["empirical_error_bound_note"]))
                    literature_notes = build_recommendation_literature_notes(target, build_orientation)
                    if literature_notes:
                        st.write("Supporting literature for recommendation:")
                        for note in literature_notes:
                            evidence = supporting_literature[supporting_literature["citation_key"] == note["citation_key"]].iloc[0]
                            st.write(f"- {evidence['display_citation']} ({evidence['doi']}): {note['note']}")
                    st.write("This route should be treated as an experimental candidate. Publication-level claims require validation on local EOS LPBF Inconel 718 specimens.")
        with st.expander("Input context retained for the dossier"):
            st.write(
                {
                    "primary_design_objective": target,
                    "initial_material_state": initial_state,
                    "representative_section_size": section_size,
                    "hip_benchmark_included": allow_hip,
                    "decision_posture": mode,
                    "available_furnace_range": furnace_limit,
                    "maximum_furnace_temperature_C": furnace_limit_C,
                    "maximum_practical_cycle_time_h": maximum_cycle_hours,
                    "surface_condition": surface_condition,
                    "build_orientation": build_orientation,
                    "cooling_condition_available": cooling_condition,
                    "target_fatigue_life_cycles": target_life_cycles,
                    "fatigue_stress_ratio_R": stress_ratio_R,
                }
            )

with tab2:
    st.subheader("Evidence Base")
    c1, c2, c3 = st.columns(3)
    c1.metric("Curated literature sources", len(sources), help="Total number of distinct papers processed in the dataset.")
    c2.metric("Local files with hashes", len(source_files), help="Number of PDFs or data files verified locally.")
    c3.metric("AM scope assessments", len(scope), help="Number of records evaluated for relevance to Additive Manufacturing and Inconel 718.")
    if not sn_targets.empty:
        st.divider()
        s1, s2, s3 = st.columns(3)
        saved_figures = int(sn_targets["digitisation_status"].astype(str).str.contains("figure_saved", na=False).sum())
        reviewed_points = int(sn_points["review_status"].astype(str).str.contains("reviewed", case=False, na=False).sum()) if "review_status" in sn_points else 0
        s1.metric("Registered S-N targets", len(sn_targets), help="Candidate or confirmed fatigue figures registered for point-level digitisation.")
        s2.metric("Saved S-N figure images", saved_figures, help="Registered targets that already have a local snipped figure image.")
        s3.metric("Reviewed S-N points", reviewed_points, help="Digitised fatigue points that have passed manual metadata review.")
        if sn_model_available:
            st.info(
                "S-N curves have been trained for source-specific literature conditions using reviewed marker points. "
                "The current recommendation system uses these curves as fatigue evidence context; route ranking remains constrained by stress ratio, surface state, and source envelope."
            )
        else:
            st.info("S-N curves have not yet been used for training; the current model uses reviewed static-property data and treats fatigue inputs as validation context.")
        if sn_audit_summary:
            a1, a2 = st.columns(2)
            a1.metric(
                "High-priority review sources",
                sn_audit_summary.get("high_priority_sources", "not available"),
                help="Sources requiring figure verification, missing snapshots, or unregistered S-N page review before digitisation.",
            )
            a2.metric(
                "Candidate fatigue pages",
                sn_audit_summary.get("fatigue_candidate_pages", "not available"),
                help="Pages flagged by broad fatigue-term screening. These are review candidates, not confirmed S-N figures.",
            )
            with st.expander("Blocking gates before fatigue model use", expanded=False):
                st.caption(
                    "Loaded from sn_digitisation_audit_summary.json. These gates must be cleared before reviewed S-N points are used for fatigue fitting."
                )
                for gate in sn_audit_summary.get("blocking_gates", []):
                    st.write(f"- {gate}")
        with st.expander("S-N digitisation register", expanded=False):
            st.caption(
                "This register tracks fatigue figures before they are used as model data. "
                "Each target retains source identifier, PDF filename, page, figure status, axis interpretation, and review status."
            )
            st.dataframe(sn_targets, width="stretch")
            st.download_button(
                "Download S-N target register",
                sn_targets.to_csv(index=False).encode("utf-8"),
                file_name="sn_digitisation_targets.csv",
                mime="text/csv",
            )
        with st.expander("Digitised S-N point data", expanded=False):
            st.caption(
                "Point rows remain empty until marker-level data are extracted and reviewed. "
                "Reviewed rows should include source, target, page, figure, curve, stress ratio, temperature, orientation, surface condition, and heat-treatment route."
            )
            if sn_points.empty:
                st.info("No reviewed S-N point rows are currently available. Registered targets must be digitised and reviewed before fatigue-life fitting.")
            else:
                st.dataframe(sn_points, width="stretch")
                st.download_button(
                    "Download reviewed S-N point table",
                    sn_points.to_csv(index=False).encode("utf-8"),
                    file_name="sn_curve_points.csv",
                    mime="text/csv",
                )
        if not sn_review_queue.empty:
            with st.expander("S-N PDF review queue", expanded=False):
                st.caption(
                    "Loaded from sn_pdf_review_queue.csv. Rows identify which source PDFs require figure verification, missing snapshot capture, or unregistered S-N page review."
                )
                st.dataframe(sn_review_queue, width="stretch")
    with st.expander("Show calibrated evidence table", expanded=False):
        raw_training = build_raw_training_data_table(sources, source_files, online_manifest)
        st.caption("The calibration evidence table includes source identifier, title, DOI, reference URL, AM-scope assessment, local file hash, and download status.")
        st.dataframe(
            raw_training,
            width="stretch",
            column_config={
                "doi": st.column_config.TextColumn("DOI"),
                "url": st.column_config.LinkColumn("Reference URL"),
                "sha256": st.column_config.TextColumn("SHA-256 file hash"),
            },
        )
        st.download_button(
            "Download calibration evidence index",
            raw_training.to_csv(index=False).encode("utf-8"),
            file_name="ht_advisor_raw_training_data_index.csv",
            mime="text/csv",
        )
    if not scope.empty:
        st.dataframe(scope, width="stretch")
    with st.expander("Supporting literature used for recommendation notes"):
        st.caption(
            "These papers support interpretation of ML fatigue methods, build-orientation effects, surface-condition sensitivity, and heat-treatment route selection. "
            "They are not direct calibration rows unless their condition-level data are later digitised and reviewed."
        )
        st.dataframe(
            supporting_literature,
            width="stretch",
            column_config={
                "url": st.column_config.LinkColumn("Reference URL"),
                "doi": st.column_config.TextColumn("DOI"),
            },
        )
    with st.expander("Source file identifiers and hashes"):
        st.dataframe(source_files, width="stretch")

with tab3:
    st.subheader("Process Window Explorer")
    if recs.empty:
        st.info("Process windows will appear after recommendation outputs are generated.")
    else:
        window_rows = build_process_window_rows(recs)
        if not window_rows.empty:
            fig = go.Figure()
            for _, row in window_rows.dropna(subset=["min_temperature_C", "max_temperature_C"]).iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row["min_temperature_C"], row["max_temperature_C"]],
                        y=[row["ht_class"], row["ht_class"]],
                        mode="lines+markers",
                        line=dict(width=8, color="#2f5d62" if row["process_family"] == "non-HIP" else "#8a6f3d"),
                        marker=dict(size=9),
                        name=row["process_family"],
                        hovertemplate=(
                            "Route: %{y}<br>"
                            "Temperature range: %{x} C<br>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )
            fig.update_xaxes(title="Observed or recommended temperature range (C)")
            fig.update_yaxes(title="Heat-treatment route")
            st.plotly_chart(academic_layout(fig, "Temperature windows represented in the current recommendation set", height=520), width="stretch")

            posture_rows = []
            if not recs.empty:
                for posture in ["conservative", "balanced", "exploratory"]:
                    subset = recs[(recs["target"] == "balanced") & (recs["allow_hip"] == False) & (recs["confidence_mode"] == posture)].copy()
                    if subset.empty:
                        continue
                    subset["decision_posture"] = posture
                    subset["recommendation_index"] = pd.to_numeric(subset["score"], errors="coerce")
                    posture_rows.append(subset)
            if posture_rows:
                animated = pd.concat(posture_rows, ignore_index=True)
                anim_fig = px.bar(
                    animated,
                    x="recommendation_index",
                    y="ht_class",
                    animation_frame="decision_posture",
                    orientation="h",
                    color="ht_class",
                    color_discrete_sequence=ACADEMIC_COLORS,
                    range_x=[0, max(0.9, float(animated["recommendation_index"].max()) + 0.05)],
                    labels={"recommendation_index": "Recommendation index", "ht_class": "Heat-treatment route"},
                )
                st.plotly_chart(academic_layout(anim_fig, "Animated sensitivity of route ranking to decision posture", height=520), width="stretch")

        cols = ["ht_class", "temperature_time_window", "inside_evidence_envelope", "confidence", "recommendation_reason"]
        st.dataframe(recs[cols].drop_duplicates(), width="stretch")

with tab4:
    st.subheader("Property Assessment")
    st.write(
        "The present release includes an empirically calibrated parametric model for available static tensile indicators only. "
        "Fatigue-life prediction remains qualitative until condition-level S-N, defect, and surface data are expanded and reviewed."
    )
    if trained_model:
        p1, p2, p3 = st.columns(3)
        p1.metric("Calibration status", trained_model.get("model_status", "not available"), help="Indicates if the property model successfully completed its empirical fitting process.")
        p2.metric("Calibration rows", trained_model.get("training_rows_total", "not available"), help="Total number of curated property measurements used to train the model.")
        p3.metric("Calibrated targets", len(trained_model.get("trained_targets", [])), help="Number of static property indicators (e.g., UTS, YS) the model can estimate.")
        st.caption(
            "The calibrated model uses heat-treatment class flags, Larson-Miller thermal dose, and Arrhenius thermal activation features. "
            "Route-level property estimates are bounded to the observed calibration-property range and reported with Empirical error bounds."
        )
    if not route_predictions.empty:
        prediction_cols = [
            c
            for c in [
                "ht_class",
                "predicted_UTS_MPa",
                "predicted_UTS_MPa_lower",
                "predicted_UTS_MPa_upper",
                "predicted_YS_MPa",
                "predicted_YS_MPa_lower",
                "predicted_YS_MPa_upper",
                "predicted_elongation_pct",
                "predicted_elongation_pct_lower",
                "predicted_elongation_pct_upper",
                "outside_training_envelope",
                "training_envelope_note",
                "empirical_error_bound_note",
                "prediction_scope",
            ]
            if c in route_predictions.columns
        ]
        pred_long = route_predictions.melt(
            id_vars=["ht_class"],
            value_vars=[c for c in ["predicted_UTS_MPa", "predicted_YS_MPa", "predicted_elongation_pct"] if c in route_predictions.columns],
            var_name="predicted_property",
            value_name="predicted_value",
        )
        if not pred_long.empty:
            pred_fig = px.bar(
                pred_long,
                x="ht_class",
                y="predicted_value",
                color="predicted_property",
                barmode="group",
                color_discrete_sequence=ACADEMIC_COLORS,
                labels={"ht_class": "Heat-treatment route", "predicted_value": "Evidence-bounded predicted value", "predicted_property": "Predicted property"},
            )
            pred_fig.update_traces(
                hovertemplate=(
                    "Route: %{x}<br>"
                    "Predicted value: %{y:.1f}<br>"
                    "Property: %{fullData.name}<br>"
                    "<extra></extra>"
                )
            )
            st.plotly_chart(academic_layout(pred_fig, "Calibrated property estimates for candidate routes", height=520), width="stretch")

            # Visualise the calibrated trade-off among strength and ductility indicators.
            if "predicted_UTS_MPa" in route_predictions.columns and "predicted_YS_MPa" in route_predictions.columns and "predicted_elongation_pct" in route_predictions.columns:
                scatter_3d = px.scatter_3d(
                    route_predictions,
                    x="predicted_UTS_MPa",
                    y="predicted_YS_MPa",
                    z="predicted_elongation_pct",
                    color="ht_class",
                    hover_name="ht_class",
                    color_discrete_sequence=ACADEMIC_COLORS,
                    labels={
                        "predicted_UTS_MPa": "UTS (MPa)",
                        "predicted_YS_MPa": "Yield Strength (MPa)",
                        "predicted_elongation_pct": "Elongation (%)",
                        "ht_class": "Route",
                    },
                )
                scatter_3d.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(academic_layout(scatter_3d, "3D Property Trade-off Landscape", height=600), width="stretch")

        if route_predictions.get("outside_training_envelope", pd.Series(dtype=bool)).astype(bool).any():
            st.warning("Extrapolation warning: at least one candidate route lies outside the reviewed calibration feature envelope. Treat its property estimate as a screening value only.")
        st.dataframe(route_predictions[prediction_cols], width="stretch")
        with st.expander("Calibration data used by the fitted property model"):
            st.dataframe(training_table, width="stretch")
    if not measurements.empty:
        counts = measurements.groupby(["ht_id", "property_type"]).size().reset_index(name="n")
        prop_fig = px.bar(
            counts,
            x="ht_id",
            y="n",
            color="property_type",
            barmode="group",
            color_discrete_sequence=ACADEMIC_COLORS,
            labels={"ht_id": "Curated heat-treatment record", "n": "Number of property records", "property_type": "Property"},
        )
        prop_fig.update_traces(
            hovertemplate=(
                "Heat-treatment record: %{x}<br>"
                "Record count: %{y}<br>"
                "Property: %{fullData.name}<br>"
                "<extra></extra>"
            )
        )
        st.plotly_chart(academic_layout(prop_fig, "Curated property evidence available for the current dataset", height=420), width="stretch")
        st.dataframe(counts, width="stretch")
        with st.expander("Curated mechanical-property records"):
            st.dataframe(measurements, width="stretch")
    else:
        st.warning("No curated mechanical-property records are currently available.")

    st.divider()
    st.markdown("#### S-N Fatigue Module")
    st.write(
        "The fatigue module uses a physics-constrained Basquin formulation fitted to reviewed literature marker points. "
        "It is presented as screening evidence for experimental planning. Not a design allowable."
    )
    if sn_model_artifact:
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Reviewed S-N points", sn_model_artifact.get("reviewed_points", len(sn_points)))
        sm2.metric("Trained S-N curves", sn_model_artifact.get("trained_condition_models", len(sn_model_summary)))
        sm3.metric("Model family", "Basquin")
        st.caption(
            "Files: sn_curve_points.csv, sn_model_summary.csv, and sn_model_prediction_grid.csv. "
            "Curves are dashed because they are literature-derived condition fits, not statistical fatigue allowables."
        )
    if sn_model_available:
        st.markdown("#### Literature S-N curves and reviewed marker points")
        plot_points = sn_points.copy()
        plot_points["cycles_to_failure"] = pd.to_numeric(plot_points["cycles_to_failure"], errors="coerce")
        plot_points["stress_amplitude_MPa"] = pd.to_numeric(plot_points["stress_amplitude_MPa"], errors="coerce")
        plot_points["runout_flag"] = plot_points["runout_flag"].astype(str)
        curve_fig = px.scatter(
            plot_points,
            x="cycles_to_failure",
            y="stress_amplitude_MPa",
            color="condition_id",
            symbol="runout_flag",
            color_discrete_sequence=ACADEMIC_COLORS,
            labels={
                "cycles_to_failure": "Cycles to failure, Nf",
                "stress_amplitude_MPa": "Stress amplitude, sigma_a (MPa)",
                "condition_id": "Condition",
                "runout_flag": "Runout flag",
            },
            custom_data=["source_id", "figure_id", "heat_treatment_class", "stress_ratio_R", "runout_flag"],
        )
        curve_fig.update_traces(
            marker=dict(size=9, line=dict(width=1, color="#17212b")),
            hovertemplate=(
                "Source: %{customdata[0]}<br>"
                "Figure: %{customdata[1]}<br>"
                "Route: %{customdata[2]}<br>"
                "R: %{customdata[3]}<br>"
                "Runout: %{customdata[4]}<br>"
                "Nf: %{x:.2e}<br>"
                "sigma_a: %{y:.1f} MPa<br>"
                "<extra></extra>"
            ),
        )
        for _, subset in sn_model_prediction_grid.groupby("condition_id"):
            line_dash_style = "dash"
            curve_fig.add_trace(
                go.Scatter(
                    x=subset["cycles_to_failure"],
                    y=subset["stress_amplitude_MPa"],
                    mode="lines",
                    name=f"{subset['condition_id'].iloc[0]} fit",
                    line=dict(width=2, dash=line_dash_style),
                    hovertemplate="Condition: %{fullData.name}<br>Nf: %{x:.2e}<br>sigma_a: %{y:.1f} MPa<extra></extra>",
                )
            )
            curve_fig.add_trace(
                go.Scatter(
                    x=pd.concat([subset["cycles_to_failure"], subset["cycles_to_failure"].iloc[::-1]]),
                    y=pd.concat([subset["stress_upper_MPa"], subset["stress_lower_MPa"].iloc[::-1]]),
                    fill="toself",
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=0.12,
                    name=f"{subset['condition_id'].iloc[0]} empirical band",
                )
            )
        curve_fig.update_xaxes(type="log", title="Cycles to failure, Nf")
        curve_fig.update_yaxes(title="Stress amplitude, sigma_a (MPa)")
        st.plotly_chart(academic_layout(curve_fig, "Literature S-N curves and reviewed marker points", height=620), width="stretch")
        st.caption(
            "Interpretation: fitted lines are dashed condition-specific Basquin regressions. "
            "Runouts are plotted but excluded from least-squares fitting. R = -1 and not-reported stress-ratio data are not pooled."
        )
        st.markdown("#### S-N model specification and traceability")
        st.dataframe(
            sn_model_summary,
            width="stretch",
            column_config={
                "source_pdf": st.column_config.TextColumn("Source PDF"),
                "model_boundary": st.column_config.TextColumn("Boundary"),
            },
        )
        st.download_button(
            "Download S-N model summary",
            sn_model_summary.to_csv(index=False).encode("utf-8"),
            file_name="sn_model_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download S-N curve grid",
            sn_model_prediction_grid.to_csv(index=False).encode("utf-8"),
            file_name="sn_model_prediction_grid.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "S-N curves have not yet been trained. No reviewed S-N point rows are currently available, or model artifacts are missing."
        )

with tab5:
    st.subheader("Validation Plan")
    st.markdown("#### Must-have experimental validation")
    st.write("Minimum experiments required before presenting a framework-recommended route as a result.")
    st.dataframe(pd.DataFrame(build_must_have_experiments("selected framework route", allow_hip=False)), width="stretch")
    st.markdown("#### Fatigue validation stress schedule")
    st.write(
        "Use these levels as a screening schedule for local fatigue validation. "
        "Life at each stress level must be measured experimentally; the fitted literature S-N curves are screening evidence, not local survival estimates."
    )
    st.dataframe(fatigue_schedule, width="stretch", column_config=FATIGUE_SCHEDULE_COLUMN_CONFIG)
    st.markdown(
        """
        1. Include an as-built baseline.
        2. Include an AMS-style standard baseline.
        3. Apply the framework-recommended non-HIP route to EOS LPBF Inconel 718 specimens.
        4. Measure hardness and tensile properties.
        5. Perform SEM/EDS microstructural assessment for Laves/Nb-rich segregation and heat-treatment response.
        6. Add fatigue screening only when specimen count and machine time allow; fatigue is defect-controlled and cannot be inferred from static tensile indicators alone.
        7. Add local results to curated data and compare measured values against empirical error bounds.
        """
    )
    st.info("The principal validation figure should compare the empirical property bound with the measured local response for the selected heat-treatment route.")

with tab6:
    st.subheader("Help and Scientific Basis")
    st.caption("How to use the tool: enter the local LPBF and heat-treatment constraints, review the ranked recommendation, inspect the calibration status, and validate the selected route experimentally.")
    for section in build_help_sections():
        st.markdown(f"#### {section['title']}")
        st.write(section["body"])
    st.markdown("#### Model calibration summary")
    if trained_model:
        st.write(f"Calibration status: **{trained_model.get('model_status', 'not available')}**")
        st.write(f"Model family: **{trained_model.get('model_family', 'not available')}**")
        st.write(f"Calibration rows: **{trained_model.get('training_rows_total', 'not available')}**")
        st.write(f"Calibrated targets: **{', '.join(trained_model.get('trained_targets', [])) or 'none'}**")
        skipped = trained_model.get("skipped_targets", {})
        if skipped:
            st.write("Targets not fitted because of limited reviewed data:")
            for target, reason in skipped.items():
                st.write(f"- {target}: {reason}")
    else:
        st.warning("No calibrated model artifact is available. Calibrate the property model before using property estimates.")
    st.markdown("#### Notation and abbreviations")
    st.write("All route abbreviations and units used in the dashboard are defined here.")
    st.dataframe(build_notation_table(), width="stretch")
    st.markdown("#### Physics used in the recommendation")
    st.write(
        "The equations below describe the feature construction and recommendation index used for interpretation. "
        "The Basquin relation is fitted only for reviewed source-specific literature marker sets and is not a design allowable for the local specimens."
    )
    for _, equation in build_equation_table().iterrows():
        st.markdown(f"**{equation['name']}**")
        st.latex(str(equation["latex"]))
        st.write(str(equation["definition"]))
    st.markdown("#### Variable definitions")
    st.dataframe(
        pd.DataFrame(
            [
                {"symbol": "P", "definition": "Larson-Miller thermal dose feature."},
                {"symbol": "T_K, T_i", "definition": "Absolute treatment temperature in kelvin."},
                {"symbol": "t_h, t_i", "definition": "Hold time in hours for a treatment step."},
                {"symbol": "C", "definition": "Larson-Miller constant used for comparative thermal exposure."},
                {"symbol": "Q", "definition": "Activation energy used in the Arrhenius-style thermal index."},
                {"symbol": "R", "definition": "Universal gas constant."},
                {"symbol": "T_ref", "definition": "Reference temperature for the thermal activation index."},
                {"symbol": "sigma_a", "definition": "Stress amplitude in the Basquin fatigue relation."},
                {"symbol": "sigma_f'", "definition": "Fatigue-strength coefficient fitted only for reviewed source-specific S-N marker sets."},
                {"symbol": "N_f", "definition": "Cycles to failure; local values must be measured for the selected specimen and stress ratio."},
                {"symbol": "b", "definition": "Basquin exponent fitted only where a reviewed condition has at least three failure points."},
                {"symbol": "s_rec", "definition": "Final recommendation index."},
                {"symbol": "s_evidence", "definition": "Evidence-weighted and constraint-adjusted route score."},
                {"symbol": "s_property", "definition": "Calibrated static-property index when sufficient reviewed data exist."},
            ]
        ),
        width="stretch",
    )
    st.write("Defect-sensitive fatigue is treated qualitatively until defect-size or surface-roughness measurements are added.")
    st.markdown("#### S-N digitisation status")
    st.write(
        "The Evidence Base tab contains the S-N digitisation register. "
        "Registered figures are traceable to source identifier, PDF filename, page, figure image, and review status. "
        "Only reviewed marker points with stress metric, test temperature, stress ratio, and heat-treatment metadata are allowed into the literature S-N fitting table."
    )
    st.markdown("#### Build-orientation coordinate convention")
    st.write(
        "Coordinate convention: X and Y lie in the build plate; Z is the build direction. "
        "A vertical fatigue specimen is aligned mainly with the z-axis, while a horizontal specimen is aligned mainly with an x-axis or y-axis direction. "
        "Mixed orientations should be recorded when the gauge section or critical feature is not aligned with a single build-axis direction."
    )
    orientation_fig = go.Figure()
    orientation_fig.add_trace(
        go.Scatter3d(
            x=[0, 1.1, None, 0, 0, None, 0, 0],
            y=[0, 0, None, 0, 1.1, None, 0, 0],
            z=[0, 0, None, 0, 0, None, 0, 1.2],
            mode="lines+markers+text",
            text=["origin", "x-axis", "", "origin", "y-axis", "", "origin", "z-axis"],
            textposition="top center",
            line=dict(width=6, color="#22535a"),
            marker=dict(size=4, color="#22535a"),
            name="Coordinate axes",
            hovertemplate="%{text}<extra></extra>",
        )
    )
    orientation_fig.add_trace(
        go.Scatter3d(
            x=[0, 1, 1, 0, 0],
            y=[0, 0, 1, 1, 0],
            z=[0, 0, 0, 0, 0],
            mode="lines",
            line=dict(width=4, color="#8a6f3d"),
            name="build plate",
            hovertemplate="Build plate: X-Y plane<extra></extra>",
        )
    )
    orientation_fig.add_trace(
        go.Scatter3d(
            x=[0.55, 0.55],
            y=[0.55, 0.55],
            z=[0, 1.0],
            mode="lines+text",
            text=["", "Z build direction"],
            textposition="top center",
            line=dict(width=8, color="#a44a3f"),
            name="Z build direction",
            hovertemplate="Vertical build direction<extra></extra>",
        )
    )
    orientation_fig.update_layout(
        scene=dict(
            xaxis_title="x-axis",
            yaxis_title="y-axis",
            zaxis_title="z-axis",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=1.5, z=1.0)),
        ),
        showlegend=True,
    )
    st.plotly_chart(academic_layout(orientation_fig, "Build plate and build-direction convention", height=420), width="stretch")
    st.markdown("#### Extrapolation warning and Empirical error bounds")
    st.write(
        "The framework flags route estimates outside the reviewed calibration envelope and reports Empirical error bounds derived from calibration residuals. "
        "These intervals are screening aids, not qualification statistics, because the current reviewed dataset is small."
    )
    st.write(
        "The model does not explicitly track delta-phase fraction, Laves-phase dissolution, precipitate morphology, or defect-size distribution; these quantities require metallographic and defect-characterisation measurements."
    )
    st.warning(
        "The current literature S-N fits do not replace local fatigue testing. "
        "Fatigue claims for non-HIP LPBF Inconel 718 require local S-N data, defect characterization, and surface-condition records."
    )
    st.markdown("#### Practical interpretation")
    st.write(
        "The tool is theoretically and manufacturingly practical as a route-selection and validation-planning aid. "
        "It is not a replacement for specimen testing. Recommended schedules should be validated on local EOS LPBF Inconel 718 specimens before any property claim is made."
    )
    st.markdown("#### Supporting literature for recommendation")
    st.write(
        "The supporting literature table includes ML-fatigue methodology, build-orientation and surface-condition fatigue evidence, and newly extracted LPBF/SLM Inconel 718 heat-treatment studies. "
        "For example, Song et al. 2025 (10.3390/ma18112604) supports the need for stress-amplitude and defect descriptors before fitted fatigue-life prediction, "
        "and Jirandehi et al. 2022 (10.1016/j.addma.2022.102661) supports retaining build orientation as a fatigue-risk variable. "
        "These sources strengthen the expert-system rationale, but they are not treated as direct calibration rows until condition-level values are curated into the training table."
    )
    st.write(
        "Extracted results: Song et al. reported GAN-RF as the strongest tested fatigue-life model "
        "(R2 = 0.975, MAE = 1.13 percent). The expanded literature set supports keeping heat-treatment route, starting state, surface condition, build orientation, and local validation as explicit parts of the recommendation."
    )
    st.dataframe(
        supporting_literature[["display_citation", "title", "doi", "url", "extracted_result", "recommendation_implication"]],
        width="stretch",
        column_config={"url": st.column_config.LinkColumn("Reference URL")},
    )
