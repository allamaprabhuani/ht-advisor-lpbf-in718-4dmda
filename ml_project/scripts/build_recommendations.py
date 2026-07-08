#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_project.ht_advisor.paths import MODEL_OUTPUT_DIR
from ml_project.ht_advisor.provenance import write_csv
from ml_project.ht_advisor.recommender import RecommendationRequest, build_route_evidence_table, rank_heat_treatments


def main() -> None:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = [
        RecommendationRequest(target="balanced", allow_hip=False, confidence_mode="balanced"),
        RecommendationRequest(target="fatigue", allow_hip=False, confidence_mode="conservative"),
        RecommendationRequest(target="strength", allow_hip=False, confidence_mode="balanced"),
        RecommendationRequest(target="ductility", allow_hip=False, confidence_mode="balanced"),
        RecommendationRequest(target="balanced", allow_hip=True, confidence_mode="balanced"),
        RecommendationRequest(target="fatigue", allow_hip=True, confidence_mode="conservative"),
        RecommendationRequest(target="ductility", allow_hip=True, confidence_mode="balanced"),
    ]
    rows = []
    for s_idx, scenario in enumerate(scenarios, start=1):
        for row in rank_heat_treatments(scenario):
            rows.append({"scenario_id": f"SCENARIO_{s_idx}", **scenario.__dict__, **row})
    write_csv(MODEL_OUTPUT_DIR / "ht_recommendations.csv", rows)
    build_route_evidence_table().to_csv(MODEL_OUTPUT_DIR / "route_evidence.csv", index=False)
    with (MODEL_OUTPUT_DIR / "ht_recommendations.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote recommendations to {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
