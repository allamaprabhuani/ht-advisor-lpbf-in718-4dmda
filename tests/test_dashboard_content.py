from pathlib import Path

from ml_project.ht_advisor.recommender import BASE_ROUTES


APP_TEXT = Path("ml_project/dashboard/app.py").read_text(encoding="utf-8")


def test_dashboard_uses_academic_framing_not_internal_language():
    banned = [
        "prototype",
        "run `python3",
        "seed measurements",
        "no recommendations found",
        "data-driven expert system",
        "ML-assisted",
        "Prediction interval",
        "dashboard-recommended",
        "app flags",
        "Trained model status",
        "Training rows",
        "Trained targets",
    ]
    lowered = APP_TEXT.lower()
    for phrase in banned:
        assert phrase.lower() not in lowered


def test_route_explanations_avoid_informal_claims():
    banned = ["proven optimum", "defensible standard", "strongest"]
    route_text = " ".join(str(v) for route in BASE_ROUTES.values() for v in route.values()).lower()
    for phrase in banned:
        assert phrase not in route_text


def test_dashboard_exposes_manual_inputs_and_auditable_model_views():
    required = [
        "Manual experimental inputs",
        "Maximum practical cycle time",
        "Generate text recommendation",
        "Show calibrated evidence table",
        "Show model specification",
        "Stochastic response considerations",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_exposes_help_and_trained_model_status():
    required = [
        "Help and Scientific Basis",
        "How to use the tool",
        "Calibration status",
        "Physics used in the recommendation",
        "not a physics-informed neural network",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_exposes_supporting_literature_for_recommendations():
    required = [
        "Supporting literature for recommendation",
        "Song et al. 2025",
        "Jirandehi et al. 2022",
        "10.3390/ma18112604",
        "10.1016/j.addma.2022.102661",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_exposes_reviewer_two_safety_warnings_and_experiments():
    required = [
        "Extrapolation warning",
        "Empirical error bounds",
        "static tensile indicators only",
        "fatigue is defect-controlled",
        "Must-have experimental validation",
        "as-built baseline",
        "AMS-style standard baseline",
        "SEM/EDS",
        "empirically calibrated parametric model",
        "Estimated furnace occupancy",
        "Metallurgical rule flags",
    ]
    for phrase in required:
        assert phrase in APP_TEXT
