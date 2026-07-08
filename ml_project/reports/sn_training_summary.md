# S-N Fatigue Model Training Summary

- Reviewed marker points: 38
- Trained condition-specific Basquin models: 5
- Stress-ratio groups: -1, not reported
- Model family: right-censored Basquin screening regression
- Runout handling: Runout markers are treated as right-censored lower-bound observations. Failure points set the least-squares Basquin slope, and runouts can shift the fitted intercept upward when needed to satisfy survival-at-runout constraints.
- Claim boundary: Not a design allowable. Curves are source-specific literature fits for screening and experimental planning.
- Application boundary: No R = 0.1 fatigue-life predictor is trained yet. Current S-N curves are condition-specific literature fits for reviewed marker points at R = -1 or with stress ratio not reported; use them for screening and experiment planning only.

The fitted curves are dashed literature-derived S-N estimates with right-censored runout checks, not statistical design allowables.
