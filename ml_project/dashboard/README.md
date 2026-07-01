# HT-Advisor Computational Tool

Run from the project root:

```bash
python3 ml_project/scripts/build_provenance_tables.py
python3 ml_project/scripts/build_curated_seed.py
python3 ml_project/scripts/build_recommendations.py
python3 -m streamlit run ml_project/dashboard/app.py
```

HT-Advisor is a research decision-support framework. It recommends heat-treatment classes and observed temperature/time windows with evidence traceability. It does not claim universal optimisation.
