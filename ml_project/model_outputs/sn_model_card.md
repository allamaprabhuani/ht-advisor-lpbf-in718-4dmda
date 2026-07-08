# S-N Fatigue Model Formal Model Card

## Model Purpose

This model reconstructs literature S-N context for LPBF Inconel 718 from reviewed digitised marker points. It supports fatigue-validation planning and visual comparison, not local fatigue-life prediction.

## Data Counts

- Reviewed digitised S-N marker points: 38 reviewed digitised S-N marker points.
- Condition-specific literature S-N fits: 5.
- Stress-ratio groups: -1, not reported.
- Local R = 0.1 predictor: not trained.
- Runout rows: retained as right-censored runout observations.

## Inputs

- Stress amplitude.
- Cycles to failure or runout cycle.
- Runout flag.
- Source identifier, PDF, page, figure, curve, and condition identifier.
- Heat-treatment class, stress ratio, surface condition, build orientation, and test temperature where reported.

## Outputs

- Condition-specific Basquin screening curve.
- Empirical stress band from fit residuals.
- Right-censored runout constraint diagnostics.
- Source and condition traceability fields for each fitted curve.

## Excluded Data

- Fitted curves are not pooled across stress ratio, surface condition, or heat-treatment state.
- Curves with fewer than three failure points are not fitted.
- Literature points with unreviewed metadata are not included.
- Local fatigue results are not included because they have not yet been generated.

## Assumptions

- Failure points define the local slope of each source-specific Basquin relation.
- Right-censored runout markers indicate survival at the plotted stress and cycle count.
- Runouts can shift the fitted intercept upward when required to satisfy survival-at-runout constraints.
- Literature S-N curves are useful for screening and experiment placement, but not for local design allowables.

## Limitations

- No local R = 0.1 fatigue-life predictor has been trained.
- Surface condition, defect population, build orientation, residual stress, and specimen geometry are incomplete for some literature points.
- Digitisation uncertainty is not yet propagated as a full statistical measurement-error model.
- The model does not establish endurance limits or qualification-grade fatigue allowables.

## Allowed Claims

- Source-specific literature S-N screening curves were reconstructed from reviewed marker points.
- Runout markers are retained as right-censored observations in the screening fit diagnostics.
- The curves can guide local fatigue stress-level selection and show the expected S-N plot form.
- The current three-specimen plan is adaptive validation planning, not statistical model validation.

## Claims Not Supported

- Local fatigue-life prediction at R = 0.1.
- Design allowable generation.
- Route qualification.
- Direct transfer of R = -1 literature life to R = 0.1 local tests.
- General fatigue prediction across all AM Inconel 718 process histories.
