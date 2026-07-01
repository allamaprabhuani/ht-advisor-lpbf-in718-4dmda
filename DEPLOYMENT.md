# HT-Advisor Deployment Notes

## Recommended Hosting Route

Use GitHub as the source repository and deploy the Streamlit application through Streamlit Community Cloud.

GitHub Pages is not suitable for this application because the dashboard is a Python/Streamlit application and requires a running Python process. GitHub Pages hosts static HTML/CSS/JavaScript only.

## Files Needed in the Repository

Include:

- `requirements.txt`
- `.streamlit/config.toml`
- `README.md`
- `ml_project/dashboard/app.py`
- `ml_project/ht_advisor/`
- `ml_project/model_outputs/`
- `ml_project/curated_data/`
- `ml_project/data/sn_digitisation_targets.csv`
- `ml_project/data/sn_curve_points.csv`
- `ml_project/extracted_data/corpus_scope_audit.csv`
- `ml_project/extracted_data/manual_download_recommendations.csv`

Avoid committing publisher PDFs or snipped publisher figures unless redistribution rights are clear. The deployed dashboard uses curated data tables, source metadata, local-file hashes, and S-N target registers, not the PDF binaries.

## Streamlit Community Cloud Setup

1. Push the project to a GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create a new app from the repository.
4. Set the main file path to:

```text
ml_project/dashboard/app.py
```

5. Confirm that `requirements.txt` is in the repository root.
6. Deploy.

## Local Run Command

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run ml_project/dashboard/app.py
```

Use the regeneration scripts only when updating curated artifacts from local PDFs:

```bash
python3 ml_project/extract_literature_data.py
python3 ml_project/scripts/build_provenance_tables.py
python3 ml_project/scripts/build_curated_seed.py
python3 ml_project/scripts/train_physics_guided_model.py
python3 ml_project/scripts/build_recommendations.py
```

## Scientific Use Statement

HT-Advisor provides evidence-guided heat-treatment recommendations for LPBF Inconel 718. The outputs are experimental candidates with traceable evidence and should be validated on local material before being treated as process specifications.

The current fatigue layer is a traceable S-N digitisation register, not a fitted fatigue-life predictor. Fatigue claims require reviewed digitised S-N points, defect characterization, surface-condition records, and local validation tests.
