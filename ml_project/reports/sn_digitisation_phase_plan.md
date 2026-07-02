# S-N Digitisation Preparation Plan

This report is generated from the local corpus and source metadata. It does not digitise figure points.

## Current Counts

- PDFs in local corpus: 36
- Registered S-N targets: 21
- Saved S-N figure images: 12
- Reviewed S-N points: 38
- High-priority review sources: 26

## Blocking Gates Before Fatigue Model Use

- stress_metric_unknown
- figure_identity_unverified
- source_page_missing
- figure_image_missing
- axis_scale_unknown

## Required Workflow

1. Confirm figure identity, panel, caption, axis scale, stress metric, units, and runout symbols.
2. Save or update missing S-N figure snapshots.
3. Digitise one curve at a time and keep calibration JSON beside the local figure.
4. Replot digitised experimental marker points and compare against the source snapshot.
5. Mark points as reviewed only after source metadata and visual QA are complete.
6. Use reviewed points for Basquin or ML fatigue modelling only after sufficient conditions and source groups are available.
