from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class ManualInputContext:
    furnace_limit_C: int | None = None
    maximum_cycle_hours: float | None = None
    section_size: str = "not specified"
    surface_condition: str = "machined"
    build_orientation: str = "vertical"
    initial_material_state: str = "EOS-like LPBF, machined"
    cooling_condition: str = "controlled furnace cooling"
    target_life_cycles: int | None = None
    stress_ratio_R: float | None = 0.1
    niobium_wt_percent: float | None = None
    aluminium_wt_percent: float | None = None
    titanium_wt_percent: float | None = None


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


def select_recommendation_subset(
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


def build_example_input_combinations() -> pd.DataFrame:
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


def _temperature_values(window: str) -> list[int]:
    return [int(value) for value in re.findall(r"(\d{3,4})\s*C", window or "")]


def _estimated_cycle_hours(ht_class: str, window: str) -> float:
    hour_values: list[float] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*h", window or "", flags=re.IGNORECASE):
        hour_values.append(float(match.group(2) or match.group(1)))
    if hour_values:
        return sum(hour_values)
    if "HIP" in ht_class:
        return 24.0
    if ht_class == "HA_ST_DA":
        return 20.0
    return 16.0


def _estimated_furnace_occupancy_hours(max_temperature_c: int | None, hold_hours: float) -> float:
    if max_temperature_c is None:
        return round(hold_hours + 2.0, 2)
    ambient_c = 25.0
    ramp_rate_c_per_min = 10.0
    ramp_hours = max(max_temperature_c - ambient_c, 0.0) / ramp_rate_c_per_min / 60.0
    controlled_cooling_allowance_h = 1.0 if max_temperature_c >= 900 else 0.5
    return round(hold_hours + ramp_hours + controlled_cooling_allowance_h, 2)


def _metallurgical_rule_flags(ht_class: str, window: str) -> str:
    temperatures = _temperature_values(window)
    max_temperature = max(temperatures) if temperatures else None
    flags: list[str] = []

    if "ST" not in ht_class and "HA" not in ht_class:
        flags.append("No explicit solution treatment is specified; Laves/Nb-rich segregation may remain influential.")
    elif max_temperature is not None and max_temperature < 980:
        flags.append("Solution-treatment temperature is below 980 C; incomplete Laves-phase dissolution remains possible.")
    elif max_temperature is not None and max_temperature > 1065:
        flags.append("The solution-treatment temperature exceeds 1065 C; grain growth and delta-phase changes should be checked metallographically.")

    if "DA" in ht_class:
        flags.append("Double ageing is expected to promote gamma-prime and gamma-double-prime precipitation strengthening.")
    if "HIP" in ht_class:
        flags.append("HIP is treated as a benchmark for internal porosity reduction, not as a local primary route.")

    return " ".join(flags) if flags else "No major metallurgical rule flag was triggered for the parsed treatment window."


def _route_effects(ht_class: str) -> str:
    effects = {
        "DA": "direct ageing can increase hardness through gamma-prime and gamma-double-prime precipitation, but residual stress, segregation, and lack-of-fusion defects may remain influential.",
        "ST_DA": "solution treatment followed by double ageing is expected to promote precipitation strengthening while reducing some solidification-related heterogeneity.",
        "CUSTOM_ST_DA": "a shortened solution-treatment and double-ageing route can be practical for thin coupon screening, but it is experimental and can leave incomplete phase transformation in higher-thermal-mass sections.",
        "HA_ST_DA": "homogenisation can reduce segregation and Laves-phase effects before ageing, but it requires a higher temperature window and tighter process control.",
        "HIP_DA": "HIP can reduce internal porosity before ageing, which may improve fatigue response when defect populations dominate failure.",
        "HIP_ST_DA": "HIP combined with solution treatment and double ageing can combine pore closure with precipitation control, but it is retained here mainly as a benchmark route.",
    }
    return effects.get(ht_class, "the selected route changes precipitation state, residual stress, and defect sensitivity depending on the applied thermal cycle.")


def apply_manual_inputs(recommendations: pd.DataFrame, context: ManualInputContext) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame(
            columns=[
                "ht_class",
                "adjusted_score",
                "adjusted_rank",
                "local_feasibility",
                "constraint_notes",
                "maximum_temperature_C",
                "estimated_cycle_hours",
                "estimated_furnace_occupancy_h",
                "metallurgical_rule_flags",
                "selected_recipe_summary",
                "fatigue_validation_context",
            ]
        )

    adjusted = recommendations.copy()
    adjusted["maximum_temperature_C"] = adjusted.apply(
        lambda row: (
            int(row["recommended_peak_temperature_C"])
            if pd.notna(row.get("recommended_peak_temperature_C"))
            else max(_temperature_values(str(row.get("temperature_time_window", ""))) or [None])
        ),
        axis=1,
    )
    adjusted["estimated_cycle_hours"] = adjusted.apply(
        lambda row: (
            float(row["recommended_total_hold_h"])
            if pd.notna(row.get("recommended_total_hold_h"))
            else _estimated_cycle_hours(str(row.get("ht_class", "")), str(row.get("temperature_time_window", "")))
        ),
        axis=1,
    )
    adjusted["estimated_furnace_occupancy_h"] = adjusted.apply(
        lambda row: _estimated_furnace_occupancy_hours(row["maximum_temperature_C"], row["estimated_cycle_hours"]),
        axis=1,
    )
    adjusted["metallurgical_rule_flags"] = adjusted.apply(
        lambda row: _metallurgical_rule_flags(str(row.get("ht_class", "")), str(row.get("temperature_time_window", ""))),
        axis=1,
    )

    adjusted_scores = []
    feasibility = []
    notes = []
    for _, row in adjusted.iterrows():
        penalty = 0.0
        row_notes: list[str] = []
        max_temp = row["maximum_temperature_C"]
        cycle_hours = row["estimated_cycle_hours"]

        if context.furnace_limit_C is not None and max_temp is not None and max_temp > context.furnace_limit_C:
            penalty += 0.18 + min((max_temp - context.furnace_limit_C) / 1000.0, 0.12)
            row_notes.append("The required maximum temperature exceeds the selected furnace range.")
        if context.maximum_cycle_hours is not None and cycle_hours > context.maximum_cycle_hours:
            penalty += 0.08
            row_notes.append("The estimated treatment duration exceeds the selected cycle-time limit.")
        if "large" in context.section_size.lower():
            penalty += 0.02
            row_notes.append("Large sections may require additional attention to thermal gradients and cooling uniformity.")
        if "as-built" in context.surface_condition.lower():
            penalty += 0.03
            row_notes.append("As-built surface condition can increase fatigue scatter even when the heat treatment is appropriate.")

        adjusted_scores.append(round(float(row.get("score", 0.0)) - penalty, 4))
        if row_notes:
            feasibility.append("limited by selected furnace range" if "furnace" in " ".join(row_notes) else "conditional under selected constraints")
            notes.append(" ".join(row_notes))
        else:
            feasibility.append("feasible under selected constraints")
            notes.append("No constraint penalties were applied.")

    adjusted["adjusted_score"] = adjusted_scores
    adjusted["local_feasibility"] = feasibility
    adjusted["constraint_notes"] = notes
    if "selected_recipe_summary" not in adjusted.columns:
        adjusted["selected_recipe_summary"] = adjusted["temperature_time_window"]
    adjusted["fatigue_validation_context"] = _fatigue_validation_context(context)
    adjusted = adjusted.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
    adjusted["adjusted_rank"] = range(1, len(adjusted) + 1)
    return adjusted


def _fatigue_validation_context(context: ManualInputContext) -> str:
    stress_ratio = f"R = {context.stress_ratio_R:g}" if context.stress_ratio_R is not None else "R not specified"
    life = f"Nf = {context.target_life_cycles:,} cycles" if context.target_life_cycles else "Nf not specified"
    return f"{stress_ratio}; {life}"


def build_fatigue_validation_schedule(
    stress_ratio_R: float = 0.1,
    target_life_cycles: int | None = 1000000,
    stress_amplitudes_MPa: list[int] | None = None,
) -> pd.DataFrame:
    amplitudes = stress_amplitudes_MPa or [300, 350, 400, 450]
    if stress_ratio_R >= 1.0:
        raise ValueError("stress_ratio_R must be less than 1.0")

    rows: list[dict[str, object]] = []
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


def generate_text_recommendation(top_row: dict | pd.Series, context: ManualInputContext) -> str:
    row = dict(top_row)
    ht_class = str(row.get("ht_class", "the selected route"))
    selected_recipe = row.get("selected_recipe_summary", row.get("temperature_time_window", "not specified"))
    fatigue_context = _fatigue_validation_context(context)
    target_cycles = (
        f" The fatigue validation target is {fatigue_context}; this should be treated as a validation target rather than an assumed outcome."
        if context.target_life_cycles
        else f" Fatigue validation context is {fatigue_context}; fatigue life should not be inferred without local S-N testing."
    )
    return (
        f"The recommended primary route for the selected constraints is {ht_class}. "
        f"The recommended validation recipe is {selected_recipe}. "
        f"The supporting literature window is {row.get('temperature_time_window', 'not specified')}. "
        f"Expected effects are that {_route_effects(ht_class)} "
        f"Local feasibility is assessed as {row.get('local_feasibility', 'not assessed')}: {row.get('constraint_notes', 'no constraint note recorded')} "
        f"The recommendation index is {float(row.get('ml_assisted_score', row.get('adjusted_score', row.get('score', 0.0)))):.2f} with {row.get('confidence', 'unreported')} evidence confidence. "
        f"Because LPBF Inconel 718 fatigue and ductility are stochastic functions of porosity, surface state, local microstructure, and build history, the route should be validated on local validation specimens before being used for publication-level property claims.{target_cycles}"
    )


def build_model_specification() -> dict[str, object]:
    return {
        "model_family": "empirically calibrated parametric model with deterministic feasibility constraints",
        "current_role": "Heat-treatment route selection for LPBF Inconel 718 before local experimental validation.",
        "inputs": [
            "primary objective: balanced, fatigue, strength, or ductility",
            "HIP benchmark inclusion",
            "decision posture: conservative, balanced, or exploratory",
            "available furnace temperature range",
            "maximum practical cycle time",
            "section size, surface condition, build orientation, and initial material state",
            "optional composition descriptors for experimental record keeping",
            "fatigue validation context: stress ratio R and target cycles to failure Nf",
        ],
        "outputs": [
            "ranked heat-treatment route",
            "recommended validation recipe with peak temperature and total hold time",
            "recommended temperature-time window",
            "fatigue validation context with stress ratio and cycles to failure",
            "estimated furnace occupancy",
            "metallurgical rule flags",
            "local feasibility classification",
            "evidence confidence and provenance summary",
            "text recommendation describing likely metallurgical effects and validation risk",
        ],
        "uncertainty_treatment": [
            "evidence count contributes to the recommendation index",
            "routes outside the selected process envelope are penalised rather than silently removed",
            "stochastic AM response is reported qualitatively until condition-level property intervals are expanded",
            "delta-phase fraction, Laves-phase dissolution, precipitate morphology, and defect-size distributions are not explicitly resolved",
            "local experimental results should be added as validation data before making final performance claims",
        ],
        "traceability": "Each literature source is retained with local filename, SHA-256 hash, AM-scope assessment, and available DOI or URL.",
    }


def build_must_have_experiments(recommended_route: str, allow_hip: bool) -> list[dict[str, str]]:
    experiments = [
        {
            "priority": "required",
            "experiment": "as-built baseline",
            "reason": "Quantifies the starting LPBF Inconel 718 response before any thermal treatment.",
        },
        {
            "priority": "required",
            "experiment": "AMS-style standard baseline",
            "reason": "Compares the framework-recommended route against a conventional solution plus double-ageing reference route.",
        },
        {
            "priority": "required",
            "experiment": f"framework-recommended route: {recommended_route}",
            "reason": "Tests the route selected by the decision-support workflow under the local furnace constraints.",
        },
        {
            "priority": "required",
            "experiment": "hardness and tensile testing",
            "reason": "Validates the static property indicators used by the current metallurgy-informed model.",
        },
        {
            "priority": "required",
            "experiment": "SEM/EDS microstructural assessment",
            "reason": "Checks Laves/Nb-rich segregation, precipitate response, and evidence that the selected heat treatment produced the intended metallurgical change.",
        },
        {
            "priority": "strongly recommended",
            "experiment": "fatigue or staircase screening if machine time is available",
            "reason": "Non-HIP LPBF fatigue is defect-controlled; static tensile indicators alone do not establish fatigue safety.",
        },
    ]
    if allow_hip:
        experiments.append(
            {
                "priority": "comparison only",
                "experiment": "HIP benchmark route",
                "reason": "Useful as a literature benchmark, but not a primary local recommendation when HIP is unavailable.",
            }
        )
    return experiments


def build_raw_training_data_table(
    sources: pd.DataFrame,
    source_files: pd.DataFrame,
    online_manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "source_id",
        "title",
        "doi",
        "url",
        "am_scope",
        "recommended_model_use",
        "filename",
        "sha256",
        "local_path",
        "download_status",
    ]
    if sources.empty:
        return pd.DataFrame(columns=columns)

    source_cols = ["source_id", "title", "doi", "url", "am_scope", "recommended_model_use"]
    table = sources.reindex(columns=source_cols).copy()
    table = table.replace(r"^\s*$", pd.NA, regex=True)
    if online_manifest is not None and not online_manifest.empty:
        manifest = online_manifest.reindex(columns=["source_id", "title", "url"]).rename(
            columns={"title": "manifest_title", "url": "manifest_url"}
        )
        manifest = manifest.replace(r"^\s*$", pd.NA, regex=True)
        table = table.merge(manifest, on="source_id", how="left")
        table["title"] = table["title"].fillna(table["manifest_title"])
        table["url"] = table["url"].fillna(table["manifest_url"])
        table = table.drop(columns=["manifest_title", "manifest_url"])

    file_cols = ["source_id", "filename", "sha256", "local_path", "download_status"]
    files = source_files.reindex(columns=file_cols) if not source_files.empty else pd.DataFrame(columns=file_cols)
    table = table.merge(files, on="source_id", how="left")
    table = table.reindex(columns=columns)
    return table.fillna("not recorded")
