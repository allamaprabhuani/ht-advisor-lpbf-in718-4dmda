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
        "strength": 0.75,
        "ductility": 0.55,
        "fatigue": 0.55,
        "cost": 0.20,
        "evidence": 2,
        "reason": "Lower-cost ageing route for comparison; residual AM defects and microstructural heterogeneity may remain influential.",
    },
    "ST_DA": {
        "window": "Solution treatment about 980-1095 C for 1-2 h, then ageing about 720 C/8 h and 620 C/8 h",
        "strength": 0.90,
        "ductility": 0.70,
        "fatigue": 0.72,
        "cost": 0.45,
        "evidence": 6,
        "reason": "Literature-supported non-HIP route that balances precipitation strengthening and ductility in LPBF Inconel 718.",
    },
    "HIP_DA": {
        "window": "HIP about 1160-1200 C followed by ageing sequence near 720 C and 620 C",
        "strength": 0.78,
        "ductility": 0.75,
        "fatigue": 0.86,
        "cost": 0.80,
        "evidence": 3,
        "reason": "Defect closure may improve fatigue response where porosity controls failure; retained as a benchmark because HIP is not locally available.",
    },
    "HIP_ST_DA": {
        "window": "HIP about 1160-1200 C, solution treatment around 954-980 C, then double ageing near 718/720 C and 620/621 C",
        "strength": 0.86,
        "ductility": 0.82,
        "fatigue": 0.92,
        "cost": 0.95,
        "evidence": 3,
        "reason": "Fatigue-oriented benchmark route when HIP access is available; retained for comparison with locally feasible non-HIP treatments.",
    },
    "HA_ST_DA": {
        "window": "Homogenisation around 1065-1100 C, solution treatment, then double ageing",
        "strength": 0.82,
        "ductility": 0.78,
        "fatigue": 0.74,
        "cost": 0.70,
        "evidence": 2,
        "reason": "Relevant where segregation and Laves-phase control are central considerations; current evidence support is comparatively narrow.",
    },
    "CUSTOM_ST_DA": {
        "window": "Short-cycle solution treatment within the observed 980-1065 C range, followed by standard double ageing near 720 C and 620 C",
        "strength": 0.84,
        "ductility": 0.68,
        "fatigue": 0.68,
        "cost": 0.38,
        "evidence": 2,
        "reason": "Locally feasible non-HIP route for validation under available furnace constraints; evidence support remains limited and requires experimental confirmation.",
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
