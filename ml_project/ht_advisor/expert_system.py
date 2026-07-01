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
    niobium_wt_percent: float | None = None
    aluminium_wt_percent: float | None = None
    titanium_wt_percent: float | None = None


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
        "CUSTOM_ST_DA": "a shortened solution-treatment and double-ageing route can be practical for local validation while retaining the main precipitation-strengthening mechanism.",
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
            ]
        )

    adjusted = recommendations.copy()
    adjusted["maximum_temperature_C"] = adjusted.apply(
        lambda row: max(_temperature_values(str(row.get("temperature_time_window", ""))) or [None]),
        axis=1,
    )
    adjusted["estimated_cycle_hours"] = adjusted.apply(
        lambda row: _estimated_cycle_hours(str(row.get("ht_class", "")), str(row.get("temperature_time_window", ""))),
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
    adjusted = adjusted.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
    adjusted["adjusted_rank"] = range(1, len(adjusted) + 1)
    return adjusted


def generate_text_recommendation(top_row: dict | pd.Series, context: ManualInputContext) -> str:
    row = dict(top_row)
    ht_class = str(row.get("ht_class", "the selected route"))
    target_cycles = (
        f" A target fatigue life near {context.target_life_cycles:,} cycles should be treated as a validation target rather than an assumed outcome."
        if context.target_life_cycles
        else ""
    )
    return (
        f"The recommended primary route for the selected constraints is {ht_class}. "
        f"The proposed window is {row.get('temperature_time_window', 'not specified')}. "
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
        ],
        "outputs": [
            "ranked heat-treatment route",
            "recommended temperature-time window",
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
