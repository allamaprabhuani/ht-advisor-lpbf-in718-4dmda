# HT-Advisor Deployment Notes

## Recommended Hosting Route

Use GitHub as the source repository and deploy the Streamlit application through Streamlit Community Cloud.

GitHub Pages is not suitable for this application because the dashboard is a Python/Streamlit application and requires a running Python process. GitHub Pages hosts static HTML/CSS/JavaScript only.

## Files Needed in the Repository

Include:

- `requirements.txt`
- `.streamlit/config.toml`
- `ml_project/dashboard/app.py`
- `ml_project/ht_advisor/`
- `ml_project/model_outputs/`
- `ml_project/curated_data/`
- `ml_project/extracted_data/corpus_scope_audit.csv`
- `ml_project/extracted_data/manual_download_recommendations.csv`

Avoid committing publisher PDFs unless redistribution rights are clear. The deployed dashboard uses curated data tables and source metadata, not the PDF binaries.

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
python3 ml_project/scripts/build_provenance_tables.py
python3 ml_project/scripts/build_curated_seed.py
python3 ml_project/scripts/build_recommendations.py
python3 -m streamlit run ml_project/dashboard/app.py
```

## Scientific Use Statement

HT-Advisor provides evidence-guided heat-treatment recommendations for LPBF Inconel 718. The outputs are experimental candidates with traceable evidence and should be validated on local material before being treated as process specifications.

