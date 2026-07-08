# HT-Advisor Computational Tool

Run from the project root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run ml_project/dashboard/app.py
```

If using the project virtual environment directly, replace `python` with `./.venv/bin/python`.

HT-Advisor is a research decision-support framework. It recommends heat-treatment classes and observed temperature/time windows with evidence traceability. It does not claim universal optimisation.

## Evidence and S-N Data

The Evidence Base tab reports curated source counts, local source-file hashes, AM-scope assessments, the S-N digitisation register, reviewed marker points, and the literature S-N screening fits.

S-N figures and fatigue sources are tracked in `ml_project/data/sn_digitisation_targets.csv`. Reviewed point-level data belong in `ml_project/data/sn_curve_points.csv`. The current reviewed table contains 38 reviewed digitised S-N marker points and supports five condition-specific literature Basquin screening fits. Each reviewed row retains source, page, figure, curve, stress ratio, temperature, surface condition, build orientation, and heat-treatment metadata where available.

PDFs and snipped figure images are local-only artifacts and are not committed to the repository.
