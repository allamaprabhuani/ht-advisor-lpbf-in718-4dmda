#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_project.ht_advisor.paths import EXTRACTED_DATA_DIR, MODEL_OUTPUT_DIR
from ml_project.ht_advisor.physics_guided_model import train_and_write_artifacts


def main() -> None:
    seed_path = EXTRACTED_DATA_DIR / "excel_seed_mechanical_dataset.csv"
    seed_rows = pd.read_csv(seed_path)
    report, predictions = train_and_write_artifacts(seed_rows, MODEL_OUTPUT_DIR)
    trained_targets = ", ".join(report["trained_targets"]) or "none"
    print(f"Calibration status: {report['model_status']}")
    print(f"Calibration rows: {report['training_rows_total']}")
    print(f"Calibrated targets: {trained_targets}")
    print(f"Wrote {len(predictions)} route predictions to {MODEL_OUTPUT_DIR / 'route_property_predictions.csv'}")


if __name__ == "__main__":
    main()
