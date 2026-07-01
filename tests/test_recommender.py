from ml_project.ht_advisor.recommender import RecommendationRequest, rank_heat_treatments


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


def test_hip_routes_remain_available_when_explicitly_enabled():
    req = RecommendationRequest(target="fatigue", allow_hip=True, confidence_mode="balanced")
    rows = rank_heat_treatments(req)
    assert any(r["ht_class"] == "HIP_ST_DA" for r in rows)
