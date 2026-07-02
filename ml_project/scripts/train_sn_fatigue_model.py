#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_project.ht_advisor.sn_fatigue import (  # noqa: E402
    SN_POINT_COLUMNS,
    build_reviewed_sn_points,
    fit_basquin_models,
    make_prediction_grid,
)


DATA = ROOT / "ml_project" / "data"
OUTPUTS = ROOT / "ml_project" / "model_outputs"
REPORTS = ROOT / "ml_project" / "reports"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    points = build_reviewed_sn_points().reindex(columns=SN_POINT_COLUMNS)
    summary = fit_basquin_models(points)
    grid = make_prediction_grid(summary)

    points.to_csv(DATA / "sn_curve_points.csv", index=False)
    summary.to_csv(OUTPUTS / "sn_model_summary.csv", index=False)
    grid.to_csv(OUTPUTS / "sn_model_prediction_grid.csv", index=False)

    report = {
        "model_family": "physics-constrained Basquin regression",
        "training_scope": "condition-specific S-N fits for reviewed literature marker points",
        "reviewed_points": int(len(points)),
        "trained_condition_models": int(len(summary)),
        "stress_ratio_groups": sorted(str(value) for value in summary["stress_ratio_R"].dropna().unique()) if not summary.empty else [],
        "runout_handling": "Runout markers are retained in sn_curve_points.csv but excluded from least-squares Basquin fitting.",
        "claim_boundary": "Not a design allowable. Curves are source-specific literature fits for screening and experimental planning.",
        "data_files": {
            "reviewed_points": "ml_project/data/sn_curve_points.csv",
            "model_summary": "ml_project/model_outputs/sn_model_summary.csv",
            "prediction_grid": "ml_project/model_outputs/sn_model_prediction_grid.csv",
        },
    }
    (OUTPUTS / "sn_model_artifact.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (REPORTS / "sn_training_summary.md").write_text(
        "\n".join(
            [
                "# S-N Fatigue Model Training Summary",
                "",
                f"- Reviewed marker points: {report['reviewed_points']}",
                f"- Trained condition-specific Basquin models: {report['trained_condition_models']}",
                f"- Stress-ratio groups: {', '.join(report['stress_ratio_groups']) or 'none'}",
                f"- Model family: {report['model_family']}",
                f"- Claim boundary: {report['claim_boundary']}",
                "",
                "The fitted curves are dashed literature-derived S-N estimates, not statistical design allowables.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
