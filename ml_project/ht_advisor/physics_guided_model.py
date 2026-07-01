from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_project.ht_advisor.features import larson_miller_dose
from ml_project.ht_advisor.recommender import BASE_ROUTES

FEATURE_COLUMNS = [
    "has_HIP",
    "has_ST",
    "has_DA",
    "max_solution_temperature_C",
    "mean_ageing_temperature_C",
    "total_time_h",
    "solution_larson_miller",
    "ageing_larson_miller",
    "thermal_activation_index",
]

TARGET_COLUMNS = ["UTS_MPa", "YS_MPa", "elongation_pct", "hardness_HV"]
MIN_ROWS_PER_TARGET = 3
RIDGE_ALPHA = 1.0


def _numbers(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [float(v) for v in re.findall(r"\d+(?:\.\d+)?", str(value))]


def _numeric_or_nan(value: Any) -> float:
    if value in ("", "-", None):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _normalise_ht_class(value: Any) -> str:
    text = str(value or "").upper().replace("+", "_").replace("-", "_")
    if "HIP" in text and "ST" in text and "DA" in text:
        return "HIP_ST_DA"
    if "HIP" in text and "DA" in text:
        return "HIP_DA"
    if "ST" in text and "DA" in text:
        return "ST_DA"
    if "DA" in text:
        return "DA"
    return text.strip("_") or "UNSPECIFIED"


def _thermal_activation_index(solution_temps: list[float], ageing_temps: list[float], times: list[float]) -> float:
    gas_constant = 8.314462618
    activation_energy_j_mol = 255_000.0
    temperatures = solution_temps + ageing_temps
    if not temperatures:
        return 0.0
    if not times:
        times = [1.0] * len(temperatures)
    total = 0.0
    for idx, temp_c in enumerate(temperatures):
        time_h = times[min(idx, len(times) - 1)]
        temp_k = temp_c + 273.15
        reference_k = 1000.0
        total += time_h * math.exp((-activation_energy_j_mol / gas_constant) * ((1.0 / temp_k) - (1.0 / reference_k)))
    return total


def _feature_row(row: pd.Series | dict[str, Any]) -> dict[str, float | str]:
    item = dict(row)
    solution_temps = _numbers(item.get("solution_temps_C"))
    ageing_temps = _numbers(item.get("ageing_temps_C"))
    times = _numbers(item.get("all_times_h"))
    ht_class = _normalise_ht_class(item.get("ht_class"))
    return {
        "source_row": item.get("source_row", ""),
        "ref_id": item.get("ref_id", ""),
        "reference_url": item.get("reference_url", ""),
        "ht_class": ht_class,
        "has_HIP": float(item.get("has_HIP", int("HIP" in ht_class)) or 0),
        "has_ST": float(item.get("has_ST", int("ST" in ht_class or "HA" in ht_class)) or 0),
        "has_DA": float(item.get("has_DA", int("DA" in ht_class)) or 0),
        "max_solution_temperature_C": max(solution_temps) if solution_temps else 0.0,
        "mean_ageing_temperature_C": float(np.mean(ageing_temps)) if ageing_temps else 0.0,
        "total_time_h": sum(times),
        "solution_larson_miller": sum(larson_miller_dose(temp, times[0] if times else 1.0) for temp in solution_temps),
        "ageing_larson_miller": sum(larson_miller_dose(temp, times[min(idx + len(solution_temps), len(times) - 1)] if times else 1.0) for idx, temp in enumerate(ageing_temps)),
        "thermal_activation_index": _thermal_activation_index(solution_temps, ageing_temps, times),
    }


def build_training_table(seed_rows: pd.DataFrame) -> pd.DataFrame:
    if seed_rows.empty:
        return pd.DataFrame(columns=["source_row", "ref_id", "reference_url", "ht_class", *FEATURE_COLUMNS, *TARGET_COLUMNS])
    rows = []
    for _, row in seed_rows.iterrows():
        features = _feature_row(row)
        for target in TARGET_COLUMNS:
            features[target] = _numeric_or_nan(row.get(target))
        if any(not math.isnan(float(features[target])) for target in TARGET_COLUMNS):
            rows.append(features)
    return pd.DataFrame(rows)


def _design_matrix(table: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[float]]]:
    x = table[FEATURE_COLUMNS].astype(float).to_numpy()
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1.0
    scaled = (x - means) / stds
    design = np.column_stack([np.ones(len(scaled)), scaled])
    scaler = {"mean": means.tolist(), "scale": stds.tolist()}
    return design, scaler


def _fit_target(table: pd.DataFrame, target: str) -> dict[str, Any] | None:
    subset = table.dropna(subset=[target]).copy()
    if len(subset) < MIN_ROWS_PER_TARGET:
        return None
    design, scaler = _design_matrix(subset)
    y = subset[target].astype(float).to_numpy()
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    fitted = design @ coef
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
    return {
        "training_rows": int(len(subset)),
        "alpha": RIDGE_ALPHA,
        "intercept": float(coef[0]),
        "coefficients": {name: float(value) for name, value in zip(FEATURE_COLUMNS, coef[1:])},
        "feature_scaler": scaler,
        "training_rmse": rmse,
        "observed_min": float(np.min(y)),
        "observed_max": float(np.max(y)),
    }


def _feature_ranges(training_table: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(training_table[column], errors="coerce").dropna()
        if values.empty:
            ranges[column] = {"min": 0.0, "max": 0.0}
        else:
            ranges[column] = {"min": float(values.min()), "max": float(values.max())}
    return ranges


def fit_physics_guided_models(training_table: pd.DataFrame) -> dict[str, Any]:
    target_models = {}
    skipped_targets = {}
    for target in TARGET_COLUMNS:
        model = _fit_target(training_table, target) if target in training_table.columns else None
        if model is None:
            available = int(training_table[target].notna().sum()) if target in training_table.columns else 0
            skipped_targets[target] = f"insufficient reviewed rows for training; available rows = {available}"
        else:
            target_models[target] = model
    return {
        "model_status": "trained" if target_models else "not trained",
        "model_family": "empirically calibrated parametric model",
        "is_physics_informed_neural_network": False,
        "training_rows_total": int(len(training_table)),
        "feature_columns": FEATURE_COLUMNS,
        "feature_ranges": _feature_ranges(training_table) if not training_table.empty else {},
        "trained_targets": list(target_models.keys()),
        "skipped_targets": skipped_targets,
        "target_models": target_models,
        "physics_equations": {
            "larson_miller_parameter": "P = T_K * (C + log10(t_h))",
            "arrhenius_thermal_activation": "A = sum[t_h * exp((-Q/R) * (1/T_K - 1/T_ref))]",
            "basquin_law": "sigma_a = sigma_f_prime * (2Nf)^b",
            "defect_sensitive_fatigue_basis": "fatigue risk increases with near-surface defects, roughness, and stress concentration; Murakami sqrt(area) treatment can be added when defect-size data are available",
        },
        "scope_statement": (
            f"The fitted model is calibrated on reviewed LPBF/SLM Inconel 718 property rows (n = {int(len(training_table))}). "
            "It estimates static tensile indicators where data are sufficient and uses fatigue physics qualitatively until condition-level S-N data are curated. "
            "The model does not explicitly resolve delta-phase fraction, Laves-phase dissolution, precipitate morphology, or defect-size distributions."
        ),
    }


def _candidate_features(ht_class: str, window: str) -> dict[str, float | str]:
    temps = _numbers(window)
    solution_temps = [temp for temp in temps if temp >= 900]
    ageing_temps = [temp for temp in temps if 500 <= temp < 900]
    times = _numbers(" ".join(re.findall(r"\d+(?:\.\d+)?\s*h", window or "")))
    row = {
        "ht_class": ht_class,
        "has_HIP": int("HIP" in ht_class),
        "has_ST": int("ST" in ht_class or "HA" in ht_class),
        "has_DA": int("DA" in ht_class),
        "solution_temps_C": ";".join(str(v) for v in solution_temps),
        "ageing_temps_C": ";".join(str(v) for v in ageing_temps),
        "all_times_h": ";".join(str(v) for v in times),
    }
    return _feature_row(row)


def _predict_one(model: dict[str, Any], features: dict[str, Any]) -> float:
    values = np.array([float(features[col]) for col in FEATURE_COLUMNS])
    means = np.array(model["feature_scaler"]["mean"], dtype=float)
    scales = np.array(model["feature_scaler"]["scale"], dtype=float)
    scaled = (values - means) / scales
    coef = np.array([model["coefficients"][col] for col in FEATURE_COLUMNS], dtype=float)
    raw = float(model["intercept"] + scaled @ coef)
    return float(np.clip(raw, model["observed_min"], model["observed_max"]))


def _prediction_interval(model: dict[str, Any], prediction: float) -> tuple[float, float]:
    rmse = float(model.get("training_rmse", 0.0))
    observed_min = float(model["observed_min"])
    observed_max = float(model["observed_max"])
    half_width = max(1.96 * rmse, 0.05 * max(observed_max - observed_min, 1.0))
    return (
        round(max(observed_min, prediction - half_width), 2),
        round(min(observed_max, prediction + half_width), 2),
    )


def _training_envelope(features: dict[str, Any], feature_ranges: dict[str, dict[str, float]]) -> tuple[bool, str]:
    outside = []
    for column in FEATURE_COLUMNS:
        if column not in feature_ranges:
            continue
        value = float(features[column])
        low = feature_ranges[column]["min"]
        high = feature_ranges[column]["max"]
        if value < low or value > high:
            outside.append(f"{column}={value:.2g} outside [{low:.2g}, {high:.2g}]")
    if outside:
        return True, "Extrapolation warning: input parameters exceed the reviewed calibration envelope; estimates are unverified extrapolations. " + "; ".join(outside)
    return False, "Within the reviewed calibration envelope for the fitted feature ranges."


def predict_candidate_routes(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ht_class, route in BASE_ROUTES.items():
        features = _candidate_features(ht_class, route["window"])
        outside, envelope_note = _training_envelope(features, report.get("feature_ranges", {}))
        pred_row: dict[str, Any] = {
            "ht_class": ht_class,
            "temperature_time_window": route["window"],
            "ml_model_status": report["model_status"],
            "prediction_scope": "bounded to observed calibration-property range",
            "outside_training_envelope": outside,
            "training_envelope_note": envelope_note,
            "empirical_error_bound_note": "Empirical error bounds are estimated from calibration residuals and are screening bounds, not qualification intervals.",
        }
        for target, model in report.get("target_models", {}).items():
            prediction = round(_predict_one(model, features), 2)
            lower, upper = _prediction_interval(model, prediction)
            pred_row[f"predicted_{target}"] = prediction
            pred_row[f"predicted_{target}_lower"] = lower
            pred_row[f"predicted_{target}_upper"] = upper
        rows.append(pred_row)
    return pd.DataFrame(rows)


def _normalised(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    low = numeric.min()
    high = numeric.max()
    if high == low:
        return pd.Series([0.5] * len(series), index=series.index)
    return (numeric - low) / (high - low)


def apply_ml_property_ranking(recommendations: pd.DataFrame, predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations.copy()
    ranked = recommendations.copy()
    if not predictions.empty and "ht_class" in predictions.columns:
        prediction_cols = [
            c
            for c in predictions.columns
            if c.startswith("predicted_")
            or c in {"outside_training_envelope", "training_envelope_note", "prediction_scope", "empirical_error_bound_note"}
        ]
        ranked = ranked.merge(predictions[["ht_class", *prediction_cols]].drop_duplicates("ht_class"), on="ht_class", how="left")

    if target == "strength" and {"predicted_UTS_MPa", "predicted_YS_MPa"}.issubset(ranked.columns):
        ranked["ml_property_index"] = 0.5 * _normalised(ranked["predicted_UTS_MPa"]) + 0.5 * _normalised(ranked["predicted_YS_MPa"])
        ranked["ml_assistance_scope"] = "calibrated tensile-property model"
    elif target == "ductility" and "predicted_elongation_pct" in ranked.columns:
        ranked["ml_property_index"] = _normalised(ranked["predicted_elongation_pct"])
        ranked["ml_assistance_scope"] = "calibrated elongation-property model"
    elif target == "balanced" and {"predicted_UTS_MPa", "predicted_YS_MPa", "predicted_elongation_pct"}.issubset(ranked.columns):
        ranked["ml_property_index"] = (
            _normalised(ranked["predicted_UTS_MPa"]) + _normalised(ranked["predicted_YS_MPa"]) + _normalised(ranked["predicted_elongation_pct"])
        ) / 3.0
        ranked["ml_assistance_scope"] = "calibrated tensile and elongation property model"
    else:
        ranked["ml_property_index"] = 0.5
        ranked["ml_assistance_scope"] = "fatigue ranking remains evidence-guided until S-N data are fitted"

    ranked["ml_property_index"] = ranked["ml_property_index"].fillna(0.5)
    ranked["ml_assisted_score"] = 0.7 * pd.to_numeric(ranked["adjusted_score"], errors="coerce").fillna(0.0) + 0.3 * ranked["ml_property_index"]
    ranked = ranked.sort_values("ml_assisted_score", ascending=False).reset_index(drop=True)
    ranked["ml_assisted_rank"] = range(1, len(ranked) + 1)
    return ranked


def train_and_write_artifacts(seed_rows: pd.DataFrame, output_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    training_table = build_training_table(seed_rows)
    report = fit_physics_guided_models(training_table)
    predictions = predict_candidate_routes(report)
    training_table.to_csv(output_path / "physics_guided_training_table.csv", index=False)
    predictions.to_csv(output_path / "route_property_predictions.csv", index=False)
    with (output_path / "physics_guided_model.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report, predictions


def build_help_sections() -> list[dict[str, str]]:
    return [
        {
            "title": "How to use the tool",
            "body": (
                "Select the property objective, local heat-treatment constraints, surface condition, and build orientation. "
                "Review the ranked route, text recommendation, local feasibility notes, and raw-data provenance before choosing a validation schedule."
            ),
        },
        {
            "title": "Calibration status",
            "body": (
                "The current model is an empirically calibrated parametric model for reviewed LPBF Inconel 718 tensile-property rows. "
                "It is not a physics-informed neural network; the available dataset is too small for a credible PINN claim."
            ),
        },
        {
            "title": "Physics used in the recommendation",
            "body": (
                "The model uses heat-treatment class flags, solution and ageing temperatures, total time, Larson-Miller thermal dose, and Arrhenius thermal activation. "
                "Basquin fatigue behaviour and defect-sensitive fatigue reasoning are used for interpretation, but not yet fitted as an S-N predictor."
            ),
        },
        {
            "title": "Correct interpretation",
            "body": (
                "The output is a candidate-selection aid for local experimental validation. "
                "It reports Empirical error bounds and extrapolation warnings where possible, but should not be interpreted as deterministic fatigue-strength evidence; local experimental validation remains required."
            ),
        },
    ]
