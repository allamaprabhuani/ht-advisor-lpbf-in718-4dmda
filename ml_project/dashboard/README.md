# HT-Advisor Computational Tool

Run from the project root:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run ml_project/dashboard/app.py
```

HT-Advisor is a research decision-support framework. It recommends heat-treatment classes and observed temperature/time windows with evidence traceability. It does not claim universal optimisation.

## Evidence and S-N Data

The Evidence Base tab reports curated source counts, local source-file hashes, AM-scope assessments, and the S-N digitisation register.

S-N figures and fatigue sources are tracked in `ml_project/data/sn_digitisation_targets.csv`. Reviewed point-level data belong in `ml_project/data/sn_curve_points.csv`. That point table remains empty until marker-level experimental points have been digitised and reviewed with source, page, figure, curve, stress ratio, temperature, surface condition, build orientation, and heat-treatment metadata.

PDFs and snipped figure images are local-only artifacts and are not committed to the repository.
