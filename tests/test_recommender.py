from ml_project.ht_advisor.recommender import (
    RecommendationRequest,
    build_route_evidence_table,
    rank_heat_treatments,
)


def test_rank_heat_treatments_returns_ranked_routes():
    req = RecommendationRequest(target="balanced", allow_hip=True, confidence_mode="balanced")
    rows = rank_heat_treatments(req)
    assert len(rows) >= 3
    assert rows[0]["rank"] == 1
    assert rows[0]["ht_class"] in {"ST_DA", "HIP_ST_DA", "HIP_DA", "DA"}
    assert "recommendation_reason" in rows[0]
    assert "selected_recipe_summary" in rows[0]
    assert "recommended_peak_temperature_C" in rows[0]
    assert "recommended_total_hold_h" in rows[0]
    assert "route_evidence_ids" in rows[0]
    assert "score_basis" in rows[0]


def test_avoid_hip_removes_hip_routes():
    req = RecommendationRequest(target="balanced", allow_hip=False, confidence_mode="conservative")
    rows = rank_heat_treatments(req)
    assert all("HIP" not in r["ht_class"] for r in rows)


def test_default_request_assumes_no_local_hip_access():
    req = RecommendationRequest()
    assert req.allow_hip is False
    rows = rank_heat_treatments(req)
    assert rows[0]["ht_class"] == "ST_DA"
    assert rows[0]["selected_recipe_summary"] == "980 C for 1 h; 720 C for 8 h; 620 C for 8 h"
    assert rows[0]["recommended_peak_temperature_C"] == 980
    assert rows[0]["recommended_total_hold_h"] == 17.0
    assert rows[0]["evidence_count_seed"] >= 10
    assert len(rows) >= 3
    assert all("HIP" not in r["ht_class"] for r in rows)


def test_custom_short_cycle_route_is_limited_to_thin_coupon_screening():
    rows = rank_heat_treatments(RecommendationRequest(target="balanced", allow_hip=False))
    custom = next(row for row in rows if row["ht_class"] == "CUSTOM_ST_DA")
    reason = custom["recommendation_reason"].lower()
    assert "thin coupon" in reason
    assert "exploratory" in reason
    assert "incomplete phase transformation" in reason


def test_hip_routes_remain_available_when_explicitly_enabled():
    req = RecommendationRequest(target="fatigue", allow_hip=True, confidence_mode="balanced")
    rows = rank_heat_treatments(req)
    assert any(r["ht_class"] == "HIP_ST_DA" for r in rows)


def test_route_evidence_table_links_scores_to_sources_and_constraints():
    evidence = build_route_evidence_table()

    assert not evidence.empty
    assert {
        "route_evidence_id",
        "ht_class",
        "source_id",
        "evidence_role",
        "temperature_time_evidence",
        "fatigue_relevance",
        "local_feasibility_logic",
        "score_component",
    }.issubset(evidence.columns)
    assert evidence["route_evidence_id"].is_unique
    assert evidence["source_id"].astype(str).str.len().gt(0).all()
    assert set(["ST_DA", "HIP_ST_DA", "DA"]).issubset(set(evidence["ht_class"]))

    rows = rank_heat_treatments(RecommendationRequest(target="balanced", allow_hip=False))
    st_da = next(row for row in rows if row["ht_class"] == "ST_DA")
    evidence_ids = st_da["route_evidence_ids"].split(";")
    assert len(evidence_ids) >= 3
    assert set(evidence_ids).issubset(set(evidence["route_evidence_id"]))
    assert "target property index" in st_da["score_basis"]
    assert "evidence bonus" in st_da["score_basis"]
    assert "local feasibility" in st_da["score_basis"]
