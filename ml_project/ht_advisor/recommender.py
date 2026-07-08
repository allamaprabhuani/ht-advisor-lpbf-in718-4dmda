from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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


ROUTE_EVIDENCE_ROWS = [
    {
        "route_evidence_id": "REV_DA_001",
        "ht_class": "DA",
        "source_id": "SRC021",
        "evidence_id": "NIST_simplified_ageing_context",
        "evidence_role": "baseline route comparator",
        "temperature_time_evidence": "Ageing near 720 C and 620 C retained as a lower-temperature comparator.",
        "fatigue_relevance": "Does not close internal pores; used mainly to expose the cost and risk trade-off against solution-treated routes.",
        "local_feasibility_logic": "Feasible when only ageing-temperature furnace capability is available.",
        "score_component": "cost advantage and baseline comparison; limited fatigue support.",
    },
    {
        "route_evidence_id": "REV_DA_002",
        "ht_class": "DA",
        "source_id": "EXCEL_REF_[5]",
        "evidence_id": "EVID_EXCEL_0005",
        "evidence_role": "ageing-step support",
        "temperature_time_evidence": "720 C for 8 h followed by furnace cooling to 620 C and 620 C hold.",
        "fatigue_relevance": "Ageing response is relevant to strength but fatigue remains defect and surface sensitive.",
        "local_feasibility_logic": "Uses lower set points than solution-treatment routes.",
        "score_component": "supports ageing sequence used across route families.",
    },
    {
        "route_evidence_id": "REV_ST_DA_001",
        "ht_class": "ST_DA",
        "source_id": "EXCEL_REF_[2]",
        "evidence_id": "EVID_EXCEL_0002",
        "evidence_role": "route-class support",
        "temperature_time_evidence": "LPBF vertical ST+DA record in curated spreadsheet evidence.",
        "fatigue_relevance": "Supports non-HIP strengthening route; fatigue still requires local S-N validation.",
        "local_feasibility_logic": "Primary local non-HIP candidate when furnace capability reaches about 980 C.",
        "score_component": "adds route-class evidence support.",
    },
    {
        "route_evidence_id": "REV_ST_DA_002",
        "ht_class": "ST_DA",
        "source_id": "EXCEL_REF_[5]",
        "evidence_id": "EVID_EXCEL_0005",
        "evidence_role": "specific recipe support",
        "temperature_time_evidence": "980 C for 1 h; 720 C for 8 h; 620 C for 8 h.",
        "fatigue_relevance": "Relevant to the planned local validation route; no local fatigue life is inferred.",
        "local_feasibility_logic": "Selected recipe stays within the 980 C local furnace constraint.",
        "score_component": "anchors selected recipe and local feasibility bonus.",
    },
    {
        "route_evidence_id": "REV_ST_DA_003",
        "ht_class": "ST_DA",
        "source_id": "NEW10",
        "evidence_id": "SN_NEW10_Fig_14",
        "evidence_role": "S-N literature context",
        "temperature_time_evidence": "Reviewed marker points for heat-treated SLM/LPBF Inconel 718 at room temperature.",
        "fatigue_relevance": "Provides source-specific R = -1 S-N context, not R = 0.1 local prediction.",
        "local_feasibility_logic": "Used as fatigue evidence context only; local fatigue testing remains mandatory.",
        "score_component": "supports fatigue-context explanation, not deterministic life scoring.",
    },
    {
        "route_evidence_id": "REV_ST_DA_004",
        "ht_class": "ST_DA",
        "source_id": "EXCEL_REF_[4]",
        "evidence_id": "EVID_EXCEL_0003",
        "evidence_role": "static-property support",
        "temperature_time_evidence": "Curated ST+DA mechanical-property row.",
        "fatigue_relevance": "Static strength is a supporting indicator only; fatigue remains defect controlled.",
        "local_feasibility_logic": "Supports retaining ST_DA as the default non-HIP route.",
        "score_component": "adds static-property support.",
    },
    {
        "route_evidence_id": "REV_HIP_DA_001",
        "ht_class": "HIP_DA",
        "source_id": "EXCEL_REF_[5]",
        "evidence_id": "EVID_EXCEL_0006",
        "evidence_role": "HIP ageing support",
        "temperature_time_evidence": "HIP at 1200 C and 150 MPa followed by 720 C and 620 C ageing.",
        "fatigue_relevance": "HIP may improve fatigue when internal porosity dominates crack initiation.",
        "local_feasibility_logic": "Benchmark only when HIP is unavailable locally.",
        "score_component": "adds fatigue potential but receives local feasibility penalty when HIP is off.",
    },
    {
        "route_evidence_id": "REV_HIP_ST_DA_001",
        "ht_class": "HIP_ST_DA",
        "source_id": "EXCEL_REF_[1]",
        "evidence_id": "EVID_EXCEL_0001",
        "evidence_role": "HIP plus solution-age support",
        "temperature_time_evidence": "1163 C HIP at 100 MPa; 954 C solution; 718/621 C ageing.",
        "fatigue_relevance": "Benchmark route combining pore closure and precipitation control.",
        "local_feasibility_logic": "Retained for comparison; not a local primary route without HIP access.",
        "score_component": "adds benchmark fatigue and ductility support.",
    },
    {
        "route_evidence_id": "REV_HIP_ST_DA_002",
        "ht_class": "HIP_ST_DA",
        "source_id": "SRC021",
        "evidence_id": "NIST_HIP1020RQSA_context",
        "evidence_role": "industrial simplification context",
        "temperature_time_evidence": "Shortened HIP and ageing strategy reported for LPBF IN718.",
        "fatigue_relevance": "Useful industrial context for post-processing burden and pore closure, not a direct local recipe.",
        "local_feasibility_logic": "Benchmark only unless local HIP route is available.",
        "score_component": "supports comparison against non-HIP local route.",
    },
    {
        "route_evidence_id": "REV_HA_ST_DA_001",
        "ht_class": "HA_ST_DA",
        "source_id": "SRC008",
        "evidence_id": "SRC008_homogenisation_context",
        "evidence_role": "homogenisation context",
        "temperature_time_evidence": "High-temperature homogenisation plus solution/ageing route family.",
        "fatigue_relevance": "Targets segregation and Laves-phase control; fatigue benefit requires local confirmation.",
        "local_feasibility_logic": "Conditional when furnace capability exceeds the selected ST_DA recipe.",
        "score_component": "supports higher-temperature comparison route.",
    },
    {
        "route_evidence_id": "REV_HA_ST_DA_002",
        "ht_class": "HA_ST_DA",
        "source_id": "EXCEL_REF_[5]",
        "evidence_id": "EVID_EXCEL_0007",
        "evidence_role": "high-temperature solution support",
        "temperature_time_evidence": "1200 C solution followed by 720 C and 620 C ageing.",
        "fatigue_relevance": "Useful for testing segregation-control hypotheses; not the first local fatigue route.",
        "local_feasibility_logic": "May exceed a 980 C local furnace limit and therefore receives feasibility penalty.",
        "score_component": "adds high-temperature evidence with process-risk penalty.",
    },
    {
        "route_evidence_id": "REV_CUSTOM_ST_DA_001",
        "ht_class": "CUSTOM_ST_DA",
        "source_id": "EXCEL_REF_[5]",
        "evidence_id": "EVID_EXCEL_0005",
        "evidence_role": "thin-coupon screening derivative",
        "temperature_time_evidence": "Shortened 980 C solution step followed by the reviewed double-ageing sequence.",
        "fatigue_relevance": "Exploratory coupon-screening route only; fatigue response must be measured locally.",
        "local_feasibility_logic": "Allowed only for low-thermal-mass coupons and flagged for incomplete transformation risk.",
        "score_component": "supports exploratory local feasibility but not component-level recommendation.",
    },
]


def build_route_evidence_table() -> pd.DataFrame:
    """Return the auditable source rows used to justify route-level scores."""

    return pd.DataFrame(ROUTE_EVIDENCE_ROWS)


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
    evidence = build_route_evidence_table()
    for ht_class, route in BASE_ROUTES.items():
        if not request.allow_hip and "HIP" in ht_class:
            continue
        uncertainty_penalty = {"conservative": 0.08, "balanced": 0.04, "exploratory": 0.00}.get(request.confidence_mode, 0.04)
        evidence_bonus = min(route["evidence"], 6) * 0.015
        cost_penalty = route["cost"] * (0.08 if request.confidence_mode != "exploratory" else 0.03)
        target_component = _target_score(route, request.target)
        score = target_component + evidence_bonus - cost_penalty - uncertainty_penalty
        route_evidence = evidence[evidence["ht_class"] == ht_class]
        route_evidence_ids = ";".join(route_evidence["route_evidence_id"].astype(str).tolist())
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
                "route_evidence_ids": route_evidence_ids,
                "score_basis": (
                    f"target property index={target_component:.3f}; evidence bonus={evidence_bonus:.3f}; "
                    f"cost penalty={cost_penalty:.3f}; uncertainty penalty={uncertainty_penalty:.3f}; "
                    "local feasibility is applied downstream from the selected furnace/HIP constraints."
                ),
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows
