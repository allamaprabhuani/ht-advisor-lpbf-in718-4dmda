# HT-Advisor for LPBF Inconel 718

HT-Advisor is a research decision-support dashboard for selecting candidate heat-treatment routes for laser powder bed fused Inconel 718. The tool ranks literature-supported routes under local constraints, presents calibrated static-property estimates where available, and records the evidence needed for later fatigue and S-N curve modelling.

The app is intended for conference and research-team discussion. It is not a process qualification tool. Recommended schedules should be validated on local EOS LPBF Inconel 718 specimens before any property claim is made.

## Current Evidence State

- Curated literature sources: 36
- Local source-file hash records: 36
- Additive-manufacturing scope assessments: 36
- Registered S-N or fatigue-figure targets: 21
- Locally saved S-N figure images: 12
- Reviewed digitised S-N points: 0

The current trained model is an empirically calibrated, physics-guided parametric model for static tensile indicators. It is not a physics-informed neural network. Fatigue is treated qualitatively until reviewed S-N points, defect descriptors, surface condition, and local fatigue-test results are added.

## Traceability

The repo separates source files, curated metadata, and model-ready data:

- `ml_project/curated_data/sources.csv` lists source titles, DOI or URL, source type, and AM relevance.
- `ml_project/curated_data/source_files.csv` records local PDF filenames and SHA-256 hashes.
- `ml_project/extracted_data/corpus_scope_audit.csv` records AM-only inclusion and exclusion decisions.
- `ml_project/source_pdfs/intake_20260701_downloads_manifest.csv` maps the July intake filenames to stable local source names and hashes.
- `ml_project/data/sn_digitisation_targets.csv` registers S-N and fatigue-figure targets before point-level digitisation.
- `ml_project/data/sn_curve_points.csv` is the reviewed point-level S-N schema. It remains empty until marker-level points are digitised and checked.

Publisher PDFs and snipped figure images are intentionally local-only and ignored by Git. The repository commits source metadata, hashes, and curated evidence tables so results can be traced without redistributing publisher files.

## Run Locally

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run ml_project/dashboard/app.py
```

For a full regeneration pass from local PDFs and curated tables:

```bash
python3 ml_project/extract_literature_data.py
python3 ml_project/scripts/build_provenance_tables.py
python3 ml_project/scripts/build_curated_seed.py
python3 ml_project/scripts/train_physics_guided_model.py
python3 ml_project/scripts/build_recommendations.py
python3 -m streamlit run ml_project/dashboard/app.py
```

## Validation

Run the repository test suite from the project root:

```bash
python3 -m pytest -q
```

The tests check recommender behavior, dashboard wording, model artifacts, evidence provenance, and the S-N digitisation schema.

## Deployment

The dashboard is deployed as a Streamlit app with `ml_project/dashboard/app.py` as the main file. See `DEPLOYMENT.md` for hosting notes.

GitHub Pages is not suitable because the app requires a Python process.

## Research Use Boundary

Appropriate claims:

- evidence-guided ranking of candidate heat-treatment classes;
- source-traceable literature and data curation;
- physics-guided static-property screening with empirical error bounds;
- documented S-N digitisation pathway for future fatigue modelling;
- local validation plan for selected heat-treatment routes.

Claims to avoid until additional data are reviewed:

- deterministic fatigue-life prediction;
- qualification of any heat-treatment route;
- fitted S-N law for the current local material;
- PINN-based fatigue prediction.
