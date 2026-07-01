from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_project.ht_advisor.dashboard_data import build_process_window_rows, build_property_tradeoff_rows
from ml_project.ht_advisor.expert_system import (
    ManualInputContext,
    apply_manual_inputs,
    build_must_have_experiments,
    build_model_specification,
    build_raw_training_data_table,
    generate_text_recommendation,
)
try:
    from ml_project.ht_advisor.expert_system import build_example_input_combinations, select_recommendation_subset
except ImportError:
    build_example_input_combinations = None
    select_recommendation_subset = None
from ml_project.ht_advisor.literature_evidence import build_recommendation_literature_notes, build_supporting_literature_table
from ml_project.ht_advisor.physics_guided_model import build_help_sections
from ml_project.ht_advisor.physics_guided_model import apply_ml_property_ranking

CURATED = ROOT / "ml_project" / "curated_data"
OUTPUTS = ROOT / "ml_project" / "model_outputs"
EXTRACTED = ROOT / "ml_project" / "extracted_data"
LITERATURE = ROOT / "ml_project" / "literature_search"

st.set_page_config(page_title="HT-Advisor", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --academic-ink: #1f2933;
        --academic-muted: #52616b;
        --academic-border: #d8dee4;
        --academic-panel: #f7f8fa;
        --academic-accent: #2f5d62;
        --academic-accent-soft: #e6eff0;
    }
    .stApp {
        background: #fbfbfa;
        color: var(--academic-ink);
    }
    [data-testid="stHeader"] {
        background: #fbfbfa;
    }
    h1, h2, h3 {
        color: var(--academic-ink);
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] {
        background: var(--academic-panel);
        border: 1px solid var(--academic-border);
        border-radius: 4px;
        padding: 0.65rem 0.75rem;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("HT-Advisor: Evidence-Guided Heat-Treatment Selection for LPBF Inconel 718")
st.caption(
    "A source-traceable decision-support system for ranking heat-treatment routes under local processing constraints. "
    "Recommendations indicate literature-supported experimental candidates and require validation on local specimens."
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


recs = load_csv(OUTPUTS / "ht_recommendations.csv")
sources = load_csv(CURATED / "sources.csv")
source_files = load_csv(CURATED / "source_files.csv")
recipes = load_csv(CURATED / "heat_treatment_recipes.csv")
measurements = load_csv(CURATED / "mechanical_measurements.csv")
scope = load_csv(EXTRACTED / "corpus_scope_audit.csv")
online_manifest = load_csv(LITERATURE / "final_online_source_manifest.csv")
trained_model = load_json(OUTPUTS / "physics_guided_model.json")
route_predictions = load_csv(OUTPUTS / "route_property_predictions.csv")
training_table = load_csv(OUTPUTS / "physics_guided_training_table.csv")
supporting_literature = build_supporting_literature_table()

ACADEMIC_COLORS = ["#2f5d62", "#5b7f95", "#8a6f3d", "#6b7280", "#a44a3f", "#7d8f69"]
FEASIBILITY_COLORS = {
    "feasible under selected constraints": "#2f5d62",
    "conditional under selected constraints": "#8a6f3d",
    "limited by selected furnace range": "#a44a3f",
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

INPUT_HELP = {
    "target": (
        "Select the property objective used to weight the route ranking. "
        "Balanced combines strength, ductility, evidence support, and feasibility. "
        "Fatigue keeps fatigue-risk modifiers visible but does not fit S-N life. "
        "Strength gives more weight to calibrated UTS and yield-strength estimates. "
        "Ductility gives more weight to calibrated elongation estimates."
    ),
    "initial_state": (
        "Initial condition of the ESOS LPBF Inconel 718 specimens before the proposed heat treatment. "
        "As-built indicates no prior thermal relief. Stress relieved indicates residual-stress reduction before the selected route. "
        "Machined indicates surface removal that can reduce roughness-driven fatigue scatter."
    ),
    "section_size": (
        "Representative section size affects thermal-gradient risk and cycle-time interpretation. "
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


def academic_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        title_font=dict(size=17, color="#1f2933"),
        paper_bgcolor="#fbfbfa",
        plot_bgcolor="#fbfbfa",
        font=dict(family="Arial", color="#1f2933", size=13),
        margin=dict(l=20, r=20, t=55, b=35),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig

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
    st.markdown("#### Required input context")
    with st.expander("Show example combinations"):
        st.markdown("#### Example input combinations")
        st.dataframe(build_example_input_combinations(), use_container_width=True)
        st.markdown("#### Available treatment routes in the reviewed evidence base")
        if recs.empty:
            st.info("The reviewed recommendation table is not available in this session.")
        else:
            route_columns = [
                "target",
                "allow_hip",
                "confidence_mode",
                "ht_class",
                "temperature_time_window",
                "confidence",
            ]
            st.dataframe(recs[route_columns].drop_duplicates(), use_container_width=True)

    left, right = st.columns(2)
    with left:
        target = st.selectbox(
            "Primary design objective",
            ["balanced", "fatigue", "strength", "ductility"],
            help=INPUT_HELP["target"],
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
    with right:
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

    st.markdown("#### Manual experimental inputs")
    furnace_defaults = {"up to 980 C": 980, "up to 1065 C": 1065, "up to 1100 C": 1100, "not specified": 1100}
    m1, m2, m3 = st.columns(3)
    with m1:
        furnace_limit_C = st.number_input(
            "Maximum furnace temperature (C)",
            min_value=600,
            max_value=1250,
            value=furnace_defaults[furnace_limit],
            step=5,
            help=INPUT_HELP["furnace_limit_C"],
        )
    with m2:
        maximum_cycle_hours = st.number_input(
            "Maximum practical cycle time (h)",
            min_value=1.0,
            max_value=96.0,
            value=20.0,
            step=1.0,
            help=INPUT_HELP["maximum_cycle_hours"],
        )
    with m3:
        target_life_cycles = st.number_input(
            "Target fatigue life, if applicable (cycles)",
            min_value=0,
            max_value=100000000,
            value=1000000,
            step=100000,
            help=INPUT_HELP["target_life_cycles"],
        )
    m4, m5 = st.columns(2)
    with m4:
        cooling_condition = st.selectbox(
            "Cooling condition available",
            ["controlled furnace cooling", "air cooling", "water quench", "not specified"],
            help=INPUT_HELP["cooling_condition"],
        )
    with m5:
        chemistry_record = st.toggle(
            "Record nominal chemistry inputs",
            value=False,
            help=INPUT_HELP["chemistry_record"],
        )
    niobium = aluminium = titanium = None
    if chemistry_record:
        c1, c2, c3 = st.columns(3)
        with c1:
            niobium = st.number_input(
                "Nb + Ta (wt.%)",
                min_value=0.0,
                max_value=8.0,
                value=5.1,
                step=0.1,
                help=INPUT_HELP["niobium"],
            )
        with c2:
            aluminium = st.number_input(
                "Al (wt.%)",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.05,
                help=INPUT_HELP["aluminium"],
            )
        with c3:
            titanium = st.number_input(
                "Ti (wt.%)",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.05,
                help=INPUT_HELP["titanium"],
            )

    manual_context = ManualInputContext(
        furnace_limit_C=int(furnace_limit_C),
        maximum_cycle_hours=float(maximum_cycle_hours),
        section_size=section_size,
        surface_condition=surface_condition,
        build_orientation=build_orientation,
        initial_material_state=initial_state,
        cooling_condition=cooling_condition,
        target_life_cycles=int(target_life_cycles) if target_life_cycles else None,
        niobium_wt_percent=niobium,
        aluminium_wt_percent=aluminium,
        titanium_wt_percent=titanium,
    )

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
    st.markdown("#### Ranked treatment routes")
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
    if filtered.empty:
        st.warning("The reviewed evidence base is empty; treatment-route ranking cannot be computed in this session.")
    else:
        if not bool(selection_status.get("exact_match", True)):
            st.warning(str(selection_status.get("selection_note", "")))
            out_of_grid_fields = selection_status.get("out_of_grid_fields", [])
            if out_of_grid_fields:
                st.markdown("#### Out-of-grid input fields")
                st.dataframe(pd.DataFrame(out_of_grid_fields), use_container_width=True)
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
                labels={
                    "ml_assisted_score": "Model-supported recommendation index",
                    "ht_class": "Heat-treatment route",
                    "local_feasibility": "Local feasibility",
                },
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(plot_rows["ht_class"].tolist())))
            st.plotly_chart(academic_layout(fig, "Model-supported heat-treatment ranking for the selected input context"), use_container_width=True)

            ev_fig = px.scatter(
                plot_rows,
                x="ml_assisted_rank",
                y="evidence_count_seed",
                size="ml_assisted_score",
                color="ht_class",
                color_discrete_sequence=ACADEMIC_COLORS,
                labels={"ml_assisted_rank": "Model-supported recommendation rank", "evidence_count_seed": "Supporting records", "ht_class": "Heat-treatment route"},
            )
            ev_fig.update_xaxes(dtick=1)
            st.plotly_chart(academic_layout(ev_fig, "Evidence support by recommendation rank"), use_container_width=True)

        top_row = adjusted.sort_values("ml_assisted_rank").iloc[0]
        if st.button("Generate text recommendation"):
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
            st.dataframe(pd.DataFrame(build_must_have_experiments(str(top_row["ht_class"]), allow_hip)), use_container_width=True)
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

        if st.button("Show model specification"):
            st.session_state["show_model_specification"] = not st.session_state.get("show_model_specification", False)
        if st.session_state.get("show_model_specification", False):
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
                cols[1].write(row["temperature_time_window"])
                cols[2].write(f"Evidence confidence: **{row['confidence']}**")
                cols[3].write(f"Supporting records: **{row['evidence_count_seed']}**")
                st.caption(row["recommendation_reason"])
                with st.expander("Rationale and evidence status"):
                    st.write(f"Evidence envelope: **{row['inside_evidence_envelope']}**")
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
                }
            )

with tab2:
    st.subheader("Evidence Base")
    c1, c2, c3 = st.columns(3)
    c1.metric("Curated literature sources", len(sources))
    c2.metric("Local files with hashes", len(source_files))
    c3.metric("AM scope assessments", len(scope))
    if st.button("Show calibrated evidence table"):
        st.session_state["show_raw_training_data"] = not st.session_state.get("show_raw_training_data", False)
    if st.session_state.get("show_raw_training_data", False):
        raw_training = build_raw_training_data_table(sources, source_files, online_manifest)
        st.caption("The calibration evidence table includes source identifier, title, DOI, reference URL, AM-scope assessment, local file hash, and download status.")
        st.dataframe(
            raw_training,
            use_container_width=True,
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
        st.dataframe(scope, use_container_width=True)
    with st.expander("Supporting literature used for recommendation notes"):
        st.caption(
            "These papers support interpretation of ML fatigue methods and build-orientation effects. "
            "They are not direct calibration rows unless their condition-level data are later digitised and reviewed."
        )
        st.dataframe(
            supporting_literature,
            use_container_width=True,
            column_config={
                "url": st.column_config.LinkColumn("Reference URL"),
                "doi": st.column_config.TextColumn("DOI"),
            },
        )
    with st.expander("Source file identifiers and hashes"):
        st.dataframe(source_files, use_container_width=True)

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
            st.plotly_chart(academic_layout(fig, "Temperature windows represented in the current recommendation set"), use_container_width=True)

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
                st.plotly_chart(academic_layout(anim_fig, "Animated sensitivity of route ranking to decision posture"), use_container_width=True)

        cols = ["ht_class", "temperature_time_window", "inside_evidence_envelope", "confidence", "recommendation_reason"]
        st.dataframe(recs[cols].drop_duplicates(), use_container_width=True)

with tab4:
    st.subheader("Property Assessment")
    st.write(
        "The present release includes an empirically calibrated parametric model for available static tensile indicators only. "
        "Fatigue-life prediction remains qualitative until condition-level S-N, defect, and surface data are expanded and reviewed."
    )
    if trained_model:
        p1, p2, p3 = st.columns(3)
        p1.metric("Calibration status", trained_model.get("model_status", "not available"))
        p2.metric("Calibration rows", trained_model.get("training_rows_total", "not available"))
        p3.metric("Calibrated targets", len(trained_model.get("trained_targets", [])))
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
            st.plotly_chart(academic_layout(pred_fig, "Calibrated property estimates for candidate routes"), use_container_width=True)
        if route_predictions.get("outside_training_envelope", pd.Series(dtype=bool)).astype(bool).any():
            st.warning("Extrapolation warning: at least one candidate route lies outside the reviewed calibration feature envelope. Treat its property estimate as a screening value only.")
        st.dataframe(route_predictions[prediction_cols], use_container_width=True)
        with st.expander("Calibration data used by the fitted property model"):
            st.dataframe(training_table, use_container_width=True)
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
        st.plotly_chart(academic_layout(prop_fig, "Curated property evidence available for the current dataset"), use_container_width=True)
        st.dataframe(counts, use_container_width=True)
        with st.expander("Curated mechanical-property records"):
            st.dataframe(measurements, use_container_width=True)
    else:
        st.warning("No curated mechanical-property records are currently available.")

with tab5:
    st.subheader("Validation Plan")
    st.markdown("#### Must-have experimental validation")
    st.write("Minimum experiments required before presenting a framework-recommended route as a result.")
    st.dataframe(pd.DataFrame(build_must_have_experiments("selected framework route", allow_hip=False)), use_container_width=True)
    st.markdown(
        """
        1. Include an as-built baseline.
        2. Include an AMS-style standard baseline.
        3. Apply the framework-recommended non-HIP route to ESOS LPBF Inconel 718 specimens.
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
    st.markdown("#### Physics used in the recommendation")
    st.write("Larson-Miller thermal dose: `P = T_K * (C + log10(t_h))`.")
    st.write("Arrhenius thermal activation: `A = sum[t_h * exp((-Q/R) * (1/T_K - 1/T_ref))]`.")
    st.write("Fatigue interpretation uses Basquin-style S-N reasoning, `sigma_a = sigma_f_prime * (2Nf)^b`, but the present dataset is not yet sufficient for a fitted fatigue-life model.")
    st.write("Defect-sensitive fatigue is treated qualitatively until defect-size or surface-roughness measurements are added.")
    st.markdown("#### Extrapolation warning and Empirical error bounds")
    st.write(
        "The framework flags route estimates outside the reviewed calibration envelope and reports Empirical error bounds derived from calibration residuals. "
        "These intervals are screening aids, not qualification statistics, because the current reviewed dataset is small."
    )
    st.write(
        "The model does not explicitly track delta-phase fraction, Laves-phase dissolution, precipitate morphology, or defect-size distribution; these quantities require metallographic and defect-characterisation measurements."
    )
    st.warning(
        "Static tensile indicators only: fatigue is defect-controlled in non-HIP LPBF Inconel 718. "
        "Fatigue claims require S-N data, defect characterization, and surface-condition records."
    )
    st.markdown("#### Practical interpretation")
    st.write(
        "The tool is theoretically and manufacturingly practical as a route-selection and validation-planning aid. "
        "It is not a replacement for specimen testing. Recommended schedules should be validated on local ESOS LPBF Inconel 718 specimens before any property claim is made."
    )
    st.markdown("#### Supporting literature for recommendation")
    st.write(
        "Song et al. 2025 (10.3390/ma18112604) is used to justify the requirement for stress-amplitude and defect descriptors before claiming fitted ML fatigue-life prediction. "
        "Jirandehi et al. 2022 (10.1016/j.addma.2022.102661) is used to justify retaining build orientation as a fatigue-risk modifier."
    )
    st.write(
        "Extracted results: Song et al. reported GAN-RF as the strongest tested fatigue-life model "
        "(R2 = 0.975, MAE = 1.13 percent), while Jirandehi et al. reported orientation-dependent fatigue damage in LB-PBF Inconel 718."
    )
    st.dataframe(
        supporting_literature[["display_citation", "title", "doi", "url", "extracted_result", "recommendation_implication"]],
        use_container_width=True,
        column_config={"url": st.column_config.LinkColumn("Reference URL")},
    )
