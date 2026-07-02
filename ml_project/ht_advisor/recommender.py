from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendationRequest:
    target: str = "balanced"
    allow_hip: bool = False
    confidence_mode: str = "balanced"
    eos_like_only: bool = False
    surface_state: str = "machined"
    build_orientation: str = "vertical"


BASE_ROUTES = {
    "DA": {
        "window": "Direct ageing around 720 C for 8 h plus 620 C for 8 h where applicable",
        "selected_recipe": "720 C for 8 h; 620 C for 8 h",
        "peak_temperature_C": 720,
        "total_hold_h": 16.0,
        "strength": 0.75,
        "ductility": 0.55,
        "fatigue": 0.55,
        "cost": 0.20,
        "evidence": 3,
        "reason": "Lower-cost ageing route for comparison; residual AM defects and microstructural heterogeneity may remain influential, so this route is best retained as a baseline rather than the primary validation route.",
    },
    "ST_DA": {
        "window": "Solution treatment about 980-1095 C for 1-2 h, then ageing about 720 C/8 h and 620 C/8 h",
        "selected_recipe": "980 C for 1 h; 720 C for 8 h; 620 C for 8 h",
        "peak_temperature_C": 980,
        "total_hold_h": 17.0,
        "strength": 0.90,
        "ductility": 0.70,
        "fatigue": 0.72,
        "cost": 0.45,
        "evidence": 12,
        "reason": "Best-supported non-HIP route in the expanded LPBF/SLM Inconel 718 evidence set; it balances precipitation strengthening, retained ductility, and practical furnace accessibility.",
    },
    "HIP_DA": {
        "window": "HIP about 1160-1200 C followed by ageing sequence near 720 C and 620 C",
        "selected_recipe": "1163 C for 3 h at 100 MPa; 720 C for 8 h; 620 C for 10 h",
        "peak_temperature_C": 1163,
        "total_hold_h": 21.0,
        "strength": 0.78,
        "ductility": 0.75,
        "fatigue": 0.86,
        "cost": 0.80,
        "evidence": 4,
        "reason": "Defect closure may improve fatigue response where porosity controls failure; retained as a benchmark because HIP is not locally available.",
    },
    "HIP_ST_DA": {
        "window": "HIP about 1160-1200 C, solution treatment around 954-980 C, then double ageing near 718/720 C and 620/621 C",
        "selected_recipe": "1163 C for 3 h at 100 MPa; 980 C for 1 h; 720 C for 8 h; 620 C for 10 h",
        "peak_temperature_C": 1163,
        "total_hold_h": 22.0,
        "strength": 0.86,
        "ductility": 0.82,
        "fatigue": 0.92,
        "cost": 0.95,
        "evidence": 4,
        "reason": "Fatigue-oriented benchmark route when HIP access is available; retained for comparison with locally feasible non-HIP treatments.",
    },
    "HA_ST_DA": {
        "window": "Homogenisation around 1065-1100 C, solution treatment, then double ageing",
        "selected_recipe": "1065 C for 1 h; 980 C for 1 h; 720 C for 8 h; 620 C for 8 h",
        "peak_temperature_C": 1065,
        "total_hold_h": 18.0,
        "strength": 0.82,
        "ductility": 0.78,
        "fatigue": 0.74,
        "cost": 0.70,
        "evidence": 5,
        "reason": "Relevant where segregation and Laves-phase control are central considerations; the expanded corpus supports homogenisation and solution-time sensitivity, but the longer furnace cycle requires local validation.",
    },
    "CUSTOM_ST_DA": {
        "window": "Short-cycle solution treatment within the observed 980-1065 C range, followed by standard double ageing near 720 C and 620 C",
        "selected_recipe": "980 C for 0.5 h; 720 C for 8 h; 620 C for 8 h",
        "peak_temperature_C": 980,
        "total_hold_h": 16.5,
        "strength": 0.84,
        "ductility": 0.68,
        "fatigue": 0.68,
        "cost": 0.38,
        "evidence": 5,
        "reason": "Exploratory thin coupon screening route for local validation under available furnace constraints. The 0.5 h solution step is not a general component recommendation; it may be suitable only for low-thermal-mass specimens and may leave incomplete phase transformation in thicker sections.",
    },
}


def _target_score(route: dict, target: str) -> float:
    if target == "fatigue":
        return 0.65 * route["fatigue"] + 0.20 * route["strength"] + 0.15 * route["ductility"]
    if target == "strength":
        return 0.65 * route["strength"] + 0.20 * route["fatigue"] + 0.15 * route["ductility"]
    if target == "ductility":
        return 0.65 * route["ductility"] + 0.20 * route["fatigue"] + 0.15 * route["strength"]
    return 0.34 * route["fatigue"] + 0.33 * route["strength"] + 0.33 * route["ductility"]


def rank_heat_treatments(request: RecommendationRequest) -> list[dict]:
    rows = []
    for ht_class, route in BASE_ROUTES.items():
        if not request.allow_hip and "HIP" in ht_class:
            continue
        uncertainty_penalty = {"conservative": 0.08, "balanced": 0.04, "exploratory": 0.00}.get(request.confidence_mode, 0.04)
        evidence_bonus = min(route["evidence"], 6) * 0.015
        cost_penalty = route["cost"] * (0.08 if request.confidence_mode != "exploratory" else 0.03)
        score = _target_score(route, request.target) + evidence_bonus - cost_penalty - uncertainty_penalty
        rows.append(
            {
                "ht_class": ht_class,
                "score": round(score, 4),
                "temperature_time_window": route["window"],
                "selected_recipe_summary": route["selected_recipe"],
                "recommended_peak_temperature_C": route["peak_temperature_C"],
                "recommended_total_hold_h": route["total_hold_h"],
                "evidence_count_seed": route["evidence"],
                "confidence": "medium" if route["evidence"] >= 3 else "low",
                "inside_evidence_envelope": "yes" if route["evidence"] >= 3 else "limited",
                "recommendation_reason": route["reason"],
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows
