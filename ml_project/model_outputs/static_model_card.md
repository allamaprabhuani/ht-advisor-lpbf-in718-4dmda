# Static-Property Model Formal Model Card

## Model Purpose

This model supports HT-Advisor by estimating static tensile indicators for candidate heat-treatment routes in LPBF Inconel 718. It is used for route screening and validation planning, not for material qualification.

## Data Counts

- Calibration rows: 6 curated heat-treatment/property records.
- The calibration rows are retained in the curated evidence tables and regenerated model outputs.
- Trained targets: UTS_MPa, YS_MPa, elongation_pct.
- Skipped targets:
- hardness_HV: insufficient reviewed rows for training; available rows = 1

## Inputs

- Heat-treatment route indicators: HIP, solution treatment, double ageing.
- Thermal recipe features: maximum solution temperature, mean ageing temperature, total hold time.
- Metallurgical dose features: solution Larson-Miller parameter, ageing Larson-Miller parameter, thermal activation index.

## Outputs

- Ultimate tensile strength estimate.
- Yield strength estimate.
- Elongation estimate.
- Empirical lower and upper bounds derived from calibration residuals.
- Training-envelope flag for extrapolative route estimates.

## Excluded Data

- Non-AM and non-Inconel 718 records are excluded from calibration.
- Background-only literature is retained for interpretation but not used as calibration rows.
- Fatigue-life data are excluded from this static-property model.

## Assumptions

- The selected route class and thermal schedule are the dominant available descriptors in the present small dataset.
- Static tensile indicators can support route screening but cannot determine fatigue resistance in defect-sensitive LPBF material.
- Empirical bounds represent calibration scatter in the curated dataset, not design allowables.

## Limitations

- The dataset is small and heterogeneous.
- Powder batch, exact scan strategy, porosity, surface roughness, build location, and residual stress are not fully available.
- The model is an empirically calibrated parametric model, not a physics-informed neural network.
- Predictions outside the reviewed feature envelope should be treated as extrapolative screening estimates.

## Allowed Claims

- Evidence-bounded screening estimates for static tensile indicators.
- Ranking support for candidate heat-treatment routes under local constraints.
- Identification of routes requiring local validation before publication-level property claims.

## Claims Not Supported

- Qualification of a heat-treatment route.
- Deterministic tensile-property certification.
- Fatigue-life prediction.
- Replacement of local tensile, hardness, microscopy, or fatigue testing.
- Physics-informed neural-network performance claims.
