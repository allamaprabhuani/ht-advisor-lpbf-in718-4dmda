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
    return notes
