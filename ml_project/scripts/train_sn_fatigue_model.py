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


def train_and_write_sn_artifacts(output_dir: str | Path = OUTPUTS, reports_dir: str | Path = REPORTS) -> dict[str, object]:
    output_path = Path(output_dir)
    reports_path = Path(reports_dir)
    DATA.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    points = build_reviewed_sn_points().reindex(columns=SN_POINT_COLUMNS)
    summary = fit_basquin_models(points)
    grid = make_prediction_grid(summary)

    points.to_csv(DATA / "sn_curve_points.csv", index=False)
    summary.to_csv(output_path / "sn_model_summary.csv", index=False)
    grid.to_csv(output_path / "sn_model_prediction_grid.csv", index=False)

    stress_ratio_groups = sorted(str(value) for value in summary["stress_ratio_R"].dropna().unique()) if not summary.empty else []
    local_r_ratio_predictor = "0.1" in stress_ratio_groups
    application_boundary = (
        "No R = 0.1 fatigue-life predictor is trained yet. Current S-N curves are condition-specific literature fits "
        "for reviewed marker points at R = -1 or with stress ratio not reported; use them for screening and experiment planning only."
    )

    report = {
        "model_family": "right-censored Basquin screening regression",
        "training_scope": "condition-specific S-N fits for reviewed literature marker points with runout lower-bound checks",
        "reviewed_points": int(len(points)),
        "trained_condition_models": int(len(summary)),
        "stress_ratio_groups": stress_ratio_groups,
        "local_r_ratio_predictor": local_r_ratio_predictor,
        "runout_handling": (
            "Runout markers are treated as right-censored lower-bound observations. Failure points set the least-squares "
            "Basquin slope, and runouts can shift the fitted intercept upward when needed to satisfy survival-at-runout constraints."
        ),
        "claim_boundary": "Not a design allowable. Curves are source-specific literature fits for screening and experimental planning.",
        "application_boundary": application_boundary,
        "data_files": {
            "reviewed_points": "ml_project/data/sn_curve_points.csv",
            "model_summary": "ml_project/model_outputs/sn_model_summary.csv",
            "prediction_grid": "ml_project/model_outputs/sn_model_prediction_grid.csv",
        },
    }
    (output_path / "sn_model_artifact.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_path / "sn_model_card.md").write_text(build_sn_model_card(report), encoding="utf-8")
    (reports_path / "sn_training_summary.md").write_text(
        "\n".join(
            [
                "# S-N Fatigue Model Training Summary",
                "",
                f"- Reviewed marker points: {report['reviewed_points']}",
                f"- Trained condition-specific Basquin models: {report['trained_condition_models']}",
                f"- Stress-ratio groups: {', '.join(report['stress_ratio_groups']) or 'none'}",
                f"- Model family: {report['model_family']}",
                f"- Runout handling: {report['runout_handling']}",
                f"- Claim boundary: {report['claim_boundary']}",
                f"- Application boundary: {report['application_boundary']}",
                "",
                "The fitted curves are dashed literature-derived S-N estimates with right-censored runout checks, not statistical design allowables.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def build_sn_model_card(report: dict[str, object]) -> str:
    stress_ratios = ", ".join(str(value) for value in report.get("stress_ratio_groups", [])) or "none"
    return (
        "# S-N Fatigue Model Formal Model Card\n\n"
        "## Model Purpose\n\n"
        "This model reconstructs literature S-N context for LPBF Inconel 718 from reviewed digitised marker points. "
        "It supports fatigue-validation planning and visual comparison, not local fatigue-life prediction.\n\n"
        "## Data Counts\n\n"
        f"- Reviewed digitised S-N marker points: {report.get('reviewed_points', 'not available')} reviewed digitised S-N marker points.\n"
        f"- Condition-specific literature S-N fits: {report.get('trained_condition_models', 'not available')}.\n"
        f"- Stress-ratio groups: {stress_ratios}.\n"
        "- Local R = 0.1 predictor: not trained.\n"
        "- Runout rows: retained as right-censored runout observations.\n\n"
        "## Inputs\n\n"
        "- Stress amplitude.\n"
        "- Cycles to failure or runout cycle.\n"
        "- Runout flag.\n"
        "- Source identifier, PDF, page, figure, curve, and condition identifier.\n"
        "- Heat-treatment class, stress ratio, surface condition, build orientation, and test temperature where reported.\n\n"
        "## Outputs\n\n"
        "- Condition-specific Basquin screening curve.\n"
        "- Empirical stress band from fit residuals.\n"
        "- Right-censored runout constraint diagnostics.\n"
        "- Source and condition traceability fields for each fitted curve.\n\n"
        "## Excluded Data\n\n"
        "- Fitted curves are not pooled across stress ratio, surface condition, or heat-treatment state.\n"
        "- Curves with fewer than three failure points are not fitted.\n"
        "- Literature points with unreviewed metadata are not included.\n"
        "- Local fatigue results are not included because they have not yet been generated.\n\n"
        "## Assumptions\n\n"
        "- Failure points define the local slope of each source-specific Basquin relation.\n"
        "- Right-censored runout markers indicate survival at the plotted stress and cycle count.\n"
        "- Runouts can shift the fitted intercept upward when required to satisfy survival-at-runout constraints.\n"
        "- Literature S-N curves are useful for screening and experiment placement, but not for local design allowables.\n\n"
        "## Limitations\n\n"
        "- No local R = 0.1 fatigue-life predictor has been trained.\n"
        "- Surface condition, defect population, build orientation, residual stress, and specimen geometry are incomplete for some literature points.\n"
        "- Digitisation uncertainty is not yet propagated as a full statistical measurement-error model.\n"
        "- The model does not establish endurance limits or qualification-grade fatigue allowables.\n\n"
        "## Allowed Claims\n\n"
        "- Source-specific literature S-N screening curves were reconstructed from reviewed marker points.\n"
        "- Runout markers are retained as right-censored observations in the screening fit diagnostics.\n"
        "- The curves can guide local fatigue stress-level selection and show the expected S-N plot form.\n"
        "- The current three-specimen plan is adaptive validation planning, not statistical model validation.\n\n"
        "## Claims Not Supported\n\n"
        "- Local fatigue-life prediction at R = 0.1.\n"
        "- Design allowable generation.\n"
        "- Route qualification.\n"
        "- Direct transfer of R = -1 literature life to R = 0.1 local tests.\n"
        "- General fatigue prediction across all AM Inconel 718 process histories.\n"
    )


def main() -> None:
    report = train_and_write_sn_artifacts(OUTPUTS, REPORTS)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
