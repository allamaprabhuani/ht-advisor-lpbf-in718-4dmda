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
        "Review" + "er 2",
        "S-N curves have not yet been used for training",
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
        "Fatigue stress ratio, R",
        "Recommended Thermal Processing Route",
        "Fatigue validation context",
        "Fatigue validation stress schedule",
        "stress_amplitude_MPa",
        "sigma_max_MPa",
        "goodman_equivalent_R_minus_1_MPa",
        "mean_stress_correction",
        "target_runout_cycles",
        "build_manual_context_from_inputs",
        "inspect.signature",
        "Text recommendation",
        "Generate text recommendation",
        "Show calibrated evidence table",
        "Show model specification",
        "Stochastic response considerations",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_input_controls_include_detailed_help_text():
    required = [
        "Select the property objective used to weight the route ranking.",
        "Initial condition of the EOS LPBF Inconel 718 specimens before the proposed heat treatment.",
        "Representative section size affects thermal-gradient risk and cycle-time interpretation.",
        "Thin coupon means a low-thermal-mass specimen, approximately up to 3 mm wall thickness or up to 5 mm gauge diameter.",
        "Surface condition is retained because fatigue response is sensitive to roughness and near-surface defects.",
        "Use this switch only when HIP is being considered as a literature benchmark.",
        "Decision posture controls how strongly the framework penalises limited evidence and local constraints.",
        "Available furnace range is used to penalise routes that exceed local equipment capability.",
        "Build orientation is retained as a fatigue-risk modifier.",
        "Coordinate convention: X and Y lie in the build plate; Z is the build direction.",
        "Maximum furnace temperature available for practical validation.",
        "Maximum total hold time that can be scheduled for a single heat-treatment route.",
        "Target fatigue life is used only as a validation objective.",
        "Stress ratio R is the minimum cyclic stress divided by the maximum cyclic stress.",
        "Cooling condition affects residual stress relief, precipitation response, and practical repeatability.",
        "Enable this only when composition values will be recorded for the experimental batch.",
        "Nb + Ta content is relevant to Laves/Nb-rich segregation and gamma-double-prime strengthening.",
        "Al contributes to gamma-prime precipitation.",
        "Ti contributes to gamma-prime precipitation and should be recorded with Al and Nb when available.",
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


def test_dashboard_exposes_scientific_warnings_and_experiments():
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
        "Total thermal-cycle duration",
        "Metallurgical rule flags",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_shows_example_combinations_and_nonempty_fallback_guidance():
    required = [
        "Example input combinations",
        "Show example combinations",
        "Available treatment routes in the reviewed evidence base",
        "The selected input combination is outside the reviewed recommendation grid",
        "Out-of-grid input fields",
        "Recommendations below use the closest available evidence subset",
        "Scientific interpretation note",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_renders_equations_and_abbreviation_guidance_professionally():
    required = [
        "Notation and abbreviations",
        "All route abbreviations and units used in the dashboard are defined here.",
        "build_notation_table",
        "build_equation_table",
        "st.latex",
        "Variable definitions",
        "Recommendation-index expression",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_uses_sidebar_inputs_and_expander_disclosure():
    required = [
        "with st.sidebar:",
        "Configuration",
        "Current input context",
        'with st.expander("Show model specification"',
        'with st.expander("Show calibrated evidence table"',
        'with st.expander("Generate text recommendation"',
        "st.divider()",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_adds_richer_academic_visualisations():
    required = [
        "Evidence workflow visual",
        "Research decision-support dossier",
        "From reviewed LPBF Inconel 718 evidence to a testable heat-treatment route",
        "Claim boundary",
        "does not certify fatigue life",
        "Literature corpus",
        "AM-only audit",
        "Route evidence",
        "S-N screening",
        "Technician validation",
        "visual-evidence-strip",
        "workflow-step",
        "@keyframes evidenceFadeIn",
        "prefers-reduced-motion",
        "S-N screening thumbnail",
        "Thermal-cycle thumbnail",
        "Recommended-route thermal cycle",
        "Property and evidence trade-off radar",
        "Recommendation-index contribution summary",
        "build_thermal_cycle_rows",
        "build_thermal_cycle_segment_rows",
        "Thermal step",
        "segment_label",
        "build_route_radar_rows",
        "build_recommendation_contribution_rows",
        "go.Scatterpolar",
        "go.Waterfall",
        "hovertemplate",
        "height=520",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_explains_build_orientation_coordinate_system():
    required = [
        "Build-orientation coordinate convention",
        "build plate",
        "Z build direction",
        "go.Scatter3d",
        "x-axis",
        "y-axis",
        "z-axis",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_exposes_sn_preparation_audit():
    required = [
        "S-N PDF review queue",
        "Outstanding checks before fatigue model use",
        "sn_pdf_review_queue.csv",
        "sn_digitisation_audit_summary.json",
        "High-priority review sources",
        "S-N curves have been trained for source-specific literature conditions",
        "No reviewed S-N point rows are currently available",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_exposes_printable_recommendation_report():
    required = [
        "Printable recommendation report",
        "Download full report",
        "Full input conditions",
        "Expected static-property estimates",
        "S-N training status",
        "S-N curves have not yet been trained",
        "Fatigue life is not predicted",
        "Fatigue validation schedule plot",
        "Validation stress schedule",
        "Process & Material Specifications",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_exposes_trained_sn_fatigue_module_without_design_allowable_claims():
    required = [
        "S-N Fatigue Module",
        "censored Basquin",
        "Literature S-N curves and reviewed marker points",
        "Not a design allowable",
        "sn_model_summary.csv",
        "sn_model_prediction_grid.csv",
        "sn_curve_points.csv",
        "No R = 0.1 fatigue-life predictor",
        "line_dash",
        "right-censored runout",
        "Stress-ratio translated screening table",
        "Goodman mean-stress translation",
        "build_stress_ratio_screening_table",
        "equivalent fully reversed",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_exposes_route_evidence_and_model_cards():
    required = [
        "Route evidence table",
        "route_evidence.csv",
        "route_evidence_ids",
        "score_basis",
        "Download route evidence table",
        "Formal model cards",
        "static_model_card.md",
        "sn_model_card.md",
        "Download static model card",
        "Download S-N model card",
    ]
    for phrase in required:
        assert phrase in APP_TEXT


def test_dashboard_print_area_exposes_technician_instruction_sheet():
    required = [
        "Print / export",
        "Print visible report",
        "window.print()",
        "Download full report",
        "Download technician sheet",
        "Technician heat-treatment instruction sheet",
        "draft work instruction for technician review",
        "Specimen or batch ID",
        "Furnace ID",
        "Furnace programme ID",
        "Nominal thermal programme",
        "Stage interpretation through the profile",
        "solution treatment is completed first",
        "Double ageing - first hold",
        "Double ageing - second hold",
        "Ramp to 980 C, then hold 980 C for 1 h",
        "After solution treatment, hold 720 C for 8 h",
        "Then hold 620 C for 8 h",
        "Final cooling method",
        "Required process records",
        "Operator sign-off",
    ]
    for phrase in required:
        assert phrase in APP_TEXT
