from __future__ import annotations

import pandas as pd


SUPPORTING_LITERATURE = [
    {
        "citation_key": "Song2025_Materials_ML_Fatigue",
        "display_citation": "Song et al. 2025",
        "title": "High-Cycle Fatigue Life Prediction of Additive Manufacturing Inconel 718 Alloy via Machine Learning",
        "doi": "10.3390/ma18112604",
        "url": "https://www.mdpi.com/1996-1944/18/11/2604",
        "use_in_app": "ML-fatigue-methodology benchmark",
        "model_use": "supporting_literature_not_training_row",
        "extracted_result": (
            "GAN-RF, a GAN-enhanced random forest model, gave the strongest reported fatigue-life prediction performance "
            "among the compared models, with R2 = 0.975 and MAE = 1.13 percent on the original test dataset. "
            "The study emphasises stress amplitude, defect characterization, defect location, and SEM-derived crack-zone descriptors."
        ),
        "recommendation_implication": (
            "Use fatigue-life ML claims only when stress amplitude, defect-size/location, and fracture or surface descriptors are available. "
            "Until those descriptors are curated locally, the dashboard should treat fatigue predictions as qualitative risk guidance."
        ),
    },
    {
        "citation_key": "Jirandehi2022_AdditiveManufacturing_BuildOrientation",
        "display_citation": "Jirandehi et al. 2022",
        "title": "Strain energy-based fatigue failure analyses of LB-PBF Inconel 718: Effect of build orientation",
        "doi": "10.1016/j.addma.2022.102661",
        "url": "https://www.osti.gov/pages/biblio/1870856",
        "use_in_app": "build-orientation fatigue-risk evidence",
        "model_use": "supporting_literature_not_training_row",
        "extracted_result": (
            "LB-PBF Inconel 718 specimens built horizontally, vertically, and at 45 degrees were tested in fully reversed bending fatigue. "
            "At a given stress level, vertical specimens endured the most fatigue damage before failure and diagonal specimens the least."
        ),
        "recommendation_implication": (
            "Build orientation should be retained as a fatigue-risk modifier. Heat treatment alone should not be presented as sufficient "
            "to remove orientation-dependent fatigue scatter."
        ),
    },
    {
        "citation_key": "Metals2021_HT_Optimisation",
        "display_citation": "Metals 2021 LPBF heat-treatment optimisation",
        "title": "Optimization of the Post-Process Heat Treatment of Inconel 718 Superalloy Fabricated by Laser Powder Bed Fusion Process",
        "doi": "10.3390/met11010144",
        "url": "https://www.mdpi.com/2075-4701/11/1/144",
        "use_in_app": "heat-treatment route evidence",
        "model_use": "supporting_literature_not_training_row",
        "extracted_result": (
            "The paper compares post-process heat-treatment routes for LPBF Inconel 718 and reports tensile, hardness, and texture-sensitive responses. "
            "It supports treating solution plus ageing schedules as the main non-HIP route family for experimental validation."
        ),
        "recommendation_implication": (
            "Use solution treatment plus ageing as the primary practical recommendation family when HIP is unavailable, while retaining hardness and tensile validation as required experiments."
        ),
    },
    {
        "citation_key": "JALCOM2022_ST_DA_Mechanical",
        "display_citation": "J. Alloys Compd. 2022 solution and double-ageing study",
        "title": "Microstructural evolution and mechanical properties of Inconel 718 alloy manufactured by selective laser melting after solution and double ageing treatments",
        "doi": "10.1016/j.jallcom.2022.164988",
        "url": "https://doi.org/10.1016/j.jallcom.2022.164988",
        "use_in_app": "solution and double-ageing evidence",
        "model_use": "supporting_literature_not_training_row",
        "extracted_result": (
            "The extracted paper is a core SLM Inconel 718 source for solution treatment followed by double ageing, with microstructure and mechanical-property evidence."
        ),
        "recommendation_implication": (
            "Strengthens the evidence basis for the ST_DA route class and motivates SEM/EDS plus tensile validation for the selected local schedule."
        ),
    },
    {
        "citation_key": "MaterChar2021_SR_ST",
        "display_citation": "Materials Characterization 2021 stress-relief and solution-treatment study",
        "title": "Laser powder bed fused Inconel 718 in stress-relieved and solution heat-treated conditions",
        "doi": "10.1016/j.matchar.2021.111499",
        "url": "https://doi.org/10.1016/j.matchar.2021.111499",
        "use_in_app": "stress-relief and solution-treatment evidence",
        "model_use": "supporting_literature_not_training_row",
        "extracted_result": (
            "The paper provides LPBF Inconel 718 evidence for stress-relieved and solution-treated conditions and links heat treatment to microstructural and mechanical response."
        ),
        "recommendation_implication": (
            "Supports retaining the initial material state and stress-relief history as input fields rather than ranking heat treatment independently of prior thermal exposure."
        ),
    },
    {
        "citation_key": "Materials2020_HA_ST_Time",
        "display_citation": "Materials 2020 homogenisation and solution-time study",
        "title": "Influence of Homogenization and Solution Treatments Time on Microstructure and Hardness of Inconel 718 Fabricated by Laser Powder Bed Fusion",
        "doi": "10.3390/ma13112574",
        "url": "https://www.mdpi.com/1996-1944/13/11/2574",
        "use_in_app": "homogenisation and solution-time evidence",
        "model_use": "supporting_literature_not_training_row",
        "extracted_result": (
            "The paper directly examines homogenisation and solution-treatment time effects for LPBF Inconel 718, with hardness and microstructure as key outputs."
        ),
        "recommendation_implication": (
            "Supports offering HA_ST_DA as a higher-temperature, longer-cycle candidate when segregation/Laves-phase control is a priority and furnace limits permit it."
        ),
    },
    {
        "citation_key": "JALCOM2022_HT_TensileFatigue",
        "display_citation": "J. Alloys Compd. 2022 tensile-fatigue heat-treatment study",
        "title": "Heat treatment for selective laser melting of Inconel 718 alloy with simultaneously enhanced tensile strength and fatigue resistance",
        "doi": "10.1016/j.jallcom.2022.165171",
        "url": "https://doi.org/10.1016/j.jallcom.2022.165171",
        "use_in_app": "heat-treatment fatigue evidence",
        "model_use": "supporting_literature_not_training_row",
        "extracted_result": (
            "The paper is a priority SLM Inconel 718 source because it links heat-treatment route selection to both tensile response and fatigue resistance."
        ),
        "recommendation_implication": (
            "Supports the expert-system direction of recommending practical heat-treatment candidates, then validating the selected route experimentally rather than inferring fatigue from tensile properties alone."
        ),
    },
    {
        "citation_key": "IFatigue2021_PostHT",
        "display_citation": "International Journal of Fatigue 2021 post-heat-treatment study",
        "title": "The microstructure and fatigue performance of Inconel 718 produced by laser-based powder bed fusion and post heat treatment",
        "doi": "10.1016/j.ijfatigue.2021.106700",
        "url": "https://doi.org/10.1016/j.ijfatigue.2021.106700",
        "use_in_app": "post-heat-treatment fatigue evidence",
        "model_use": "supporting_literature_not_training_row",
        "extracted_result": (
            "The paper connects laser PBF Inconel 718 post-heat-treatment microstructure with fatigue performance, making it a priority source for future S-N curation."
        ),
        "recommendation_implication": (
            "Reinforces the dashboard warning that heat treatment can change fatigue response but must be validated with local fatigue specimens and microstructural evidence."
        ),
    },
    {
        "citation_key": "Witkin2020_SurfaceOrientationFatigue",
        "display_citation": "Witkin et al. 2020",
        "title": "Influence of surface conditions and specimen orientation on high cycle fatigue properties of Inconel 718 prepared by laser powder bed fusion",
        "doi": "10.1016/j.ijfatigue.2019.105392",
        "url": "https://doi.org/10.1016/j.ijfatigue.2019.105392",
        "use_in_app": "surface-condition and orientation fatigue evidence",
        "model_use": "supporting_literature_not_training_row",
        "extracted_result": (
            "The extracted paper supports treating surface condition and specimen orientation as fatigue-risk variables for LPBF Inconel 718."
        ),
        "recommendation_implication": (
            "Surface condition and orientation should remain explicit user inputs; heat treatment should not be presented as sufficient to remove fatigue scatter caused by roughness or orientation."
        ),
    },
]


def build_supporting_literature_table() -> pd.DataFrame:
    return pd.DataFrame(SUPPORTING_LITERATURE)


def build_recommendation_literature_notes(target: str, build_orientation: str) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    target_l = target.lower()
    orientation_l = build_orientation.lower()
    if target_l == "fatigue":
        notes.append(
            {
                "citation_key": "Song2025_Materials_ML_Fatigue",
                "note": (
                    "Song et al. 2025 supports the use of stress amplitude and defect descriptors for AM Inconel 718 fatigue-life ML. "
                    "The present dashboard therefore does not report a fitted fatigue-life prediction until local S-N and defect data are curated."
                ),
            }
        )
        notes.append(
            {
                "citation_key": "JALCOM2022_HT_TensileFatigue",
                "note": (
                    "The 2022 heat-treatment study supports ranking solution-plus-ageing route families as fatigue-relevant candidates, "
                    "but fatigue validation still requires local S-N data and defect/surface records."
                ),
            }
        )
        notes.append(
            {
                "citation_key": "IFatigue2021_PostHT",
                "note": (
                    "The 2021 post-heat-treatment fatigue study supports coupling fatigue interpretation to microstructure and heat-treatment state, "
                    "not to tensile strength alone."
                ),
            }
        )
    if target_l in {"balanced", "strength", "ductility"}:
        notes.append(
            {
                "citation_key": "Metals2021_HT_Optimisation",
                "note": (
                    "The 2021 LPBF heat-treatment optimisation source supports using solution treatment plus ageing as the main non-HIP route family for tensile and hardness validation."
                ),
            }
        )
        notes.append(
            {
                "citation_key": "JALCOM2022_ST_DA_Mechanical",
                "note": (
                    "The 2022 solution and double-ageing source strengthens the evidence basis for ST_DA route selection in SLM/LPBF Inconel 718."
                ),
            }
        )
    if target_l in {"balanced", "strength", "ductility"} and orientation_l in {"not specified", ""}:
        notes.append(
            {
                "citation_key": "MaterChar2021_SR_ST",
                "note": (
                    "Stress-relief and solution-treatment history should be retained in the input context because the starting material state affects route interpretation."
                ),
            }
        )
    if target_l in {"balanced", "strength", "ductility"}:
        notes.append(
            {
                "citation_key": "Materials2020_HA_ST_Time",
                "note": (
                    "Homogenisation and solution-treatment time evidence supports HA_ST_DA as a higher-temperature candidate when segregation control is the objective and furnace limits permit it."
                ),
            }
        )
    if target_l == "fatigue" or orientation_l not in {"not specified", ""}:
        notes.append(
            {
                "citation_key": "Jirandehi2022_AdditiveManufacturing_BuildOrientation",
                "note": (
                    "Jirandehi et al. 2022 supports retaining build orientation as a fatigue-risk variable because fatigue damage differed across "
                    "horizontal, vertical, and 45 degree LB-PBF Inconel 718 specimens."
                ),
            }
        )
        notes.append(
            {
                "citation_key": "Witkin2020_SurfaceOrientationFatigue",
                "note": (
                    "Witkin et al. 2020 supports retaining surface condition and specimen orientation as fatigue-risk modifiers for LPBF Inconel 718."
                ),
            }
        )
    return notes
