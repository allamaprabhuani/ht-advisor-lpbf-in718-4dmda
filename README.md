# HT-Advisor for LPBF Inconel 718

HT-Advisor is a research decision-support dashboard for selecting candidate heat-treatment routes for laser powder bed fused Inconel 718. The tool ranks literature-supported routes under local constraints, presents calibrated static-property estimates where available, and records the evidence needed for later fatigue and S-N curve modelling.

The app is intended for conference and research-team discussion. It is not a process qualification tool. Recommended schedules should be validated on local EOS LPBF Inconel 718 specimens before any property claim is made.

## Current Evidence State

- Curated literature sources: 36
- Local source-file hash records: 36
- Additive-manufacturing scope assessments: 36
- Registered S-N or fatigue-figure targets: 21
- Locally saved S-N figure images: 12
- Reviewed digitised S-N points: 38
- Trained condition-specific right-censored literature S-N screening fits: 5

The current trained models are empirical, physics-guided parametric models: one layer for static tensile indicators, and one condition-specific right-censored Basquin screening layer for reviewed literature S-N marker points. These are not physics-informed neural networks. The S-N fits are literature-derived screening curves at `R = -1` or with stress ratio not reported; no local `R = 0.1` fatigue-life predictor has been trained yet.

## Traceability

The repo separates source files, curated metadata, and model-ready data:

- `ml_project/curated_data/sources.csv` lists source titles, DOI or URL, source type, and AM relevance.
- `ml_project/curated_data/source_files.csv` records local PDF filenames and SHA-256 hashes.
- `ml_project/extracted_data/corpus_scope_audit.csv` records AM-only inclusion and exclusion decisions.
- `ml_project/source_pdfs/intake_20260701_downloads_manifest.csv` maps the July intake filenames to stable local source names and hashes.
- `ml_project/data/sn_digitisation_targets.csv` registers S-N and fatigue-figure targets before point-level digitisation.
- `ml_project/data/sn_curve_points.csv` is the reviewed point-level S-N table.
- `ml_project/model_outputs/route_evidence.csv` links route recommendations to source rows, temperature/time evidence, fatigue relevance, local feasibility logic, and score components.
- `ml_project/model_outputs/static_model_card.md` and `ml_project/model_outputs/sn_model_card.md` define model data counts, excluded data, assumptions, limitations, and allowed claims.
- `ml_project/model_outputs/sn_model_summary.csv`, `ml_project/model_outputs/sn_model_prediction_grid.csv`, and `ml_project/model_outputs/sn_model_artifact.json` record the trained literature S-N screening fits and their application boundary.

Publisher PDFs and snipped figure images are intentionally local-only and ignored by Git. The repository commits source metadata, hashes, and curated evidence tables so results can be traced without redistributing publisher files.

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run ml_project/dashboard/app.py
```

If using the project virtual environment directly, replace `python` with `./.venv/bin/python`.

For a full regeneration pass from local PDFs and curated tables:

```bash
python ml_project/extract_literature_data.py
python ml_project/scripts/build_provenance_tables.py
python ml_project/scripts/build_curated_seed.py
python ml_project/scripts/train_physics_guided_model.py
python ml_project/scripts/train_sn_fatigue_model.py
python ml_project/scripts/build_recommendations.py
python -m streamlit run ml_project/dashboard/app.py
```

## Validation

Run the repository test suite from the project root:

```bash
python -m pytest -q
```

The tests check recommender behavior, dashboard wording, model artifacts, evidence provenance, and the S-N digitisation schema.

## Deployment

The dashboard is deployed as a Streamlit app with `ml_project/dashboard/app.py` as the main file. See `DEPLOYMENT.md` for hosting notes.

GitHub Pages is not suitable because the app requires a Python process.

## Research Use Boundary

Appropriate claims:

- evidence-guided ranking of candidate heat-treatment classes;
- auditable route-evidence mapping for recommendation scores;
- source-traceable literature and data curation;
- physics-guided static-property screening with empirical error bounds;
- source-traceable S-N digitisation and literature-derived right-censored Basquin screening curves;
- local validation plan for selected heat-treatment routes.

Claims to avoid until additional data are reviewed:

- deterministic fatigue-life prediction;
- qualification of any heat-treatment route;
- fitted S-N law for the current local material;
- local `R = 0.1` fatigue-life prediction before local fatigue data are available;
- PINN-based fatigue prediction.
