from __future__ import annotations

import re

import pandas as pd


def parse_process_window(ht_class: str, window: str) -> dict[str, object]:
    temps = [int(v) for v in re.findall(r"(\d{3,4})\s*C", window or "")]
    return {
        "ht_class": ht_class,
        "min_temperature_C": min(temps) if temps else None,
        "max_temperature_C": max(temps) if temps else None,
        "process_family": "HIP benchmark" if "HIP" in ht_class else "non-HIP",
        "temperature_time_window": window,
    }


def build_process_window_rows(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame(columns=["ht_class", "min_temperature_C", "max_temperature_C", "process_family", "temperature_time_window"])
    rows = [
        parse_process_window(str(row["ht_class"]), str(row["temperature_time_window"]))
        for _, row in recommendations[["ht_class", "temperature_time_window"]].drop_duplicates().iterrows()
    ]
    return pd.DataFrame(rows)


def build_property_tradeoff_rows(recommendations: pd.DataFrame, target: str, allow_hip: bool, confidence_mode: str) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame(columns=["ht_class", "rank", "recommendation_index", "evidence_count", "confidence"])
    filtered = recommendations[
        (recommendations["target"] == target)
        & (recommendations["allow_hip"] == allow_hip)
        & (recommendations["confidence_mode"] == confidence_mode)
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["ht_class", "rank", "recommendation_index", "evidence_count", "confidence"])
    filtered["recommendation_index"] = pd.to_numeric(filtered["score"], errors="coerce")
    filtered["evidence_count"] = pd.to_numeric(filtered["evidence_count_seed"], errors="coerce")
    if "confidence" not in filtered.columns:
        filtered["confidence"] = ""
    return filtered.sort_values("rank")[["ht_class", "rank", "recommendation_index", "evidence_count", "confidence"]]


def _temperature_time_steps(window: str) -> list[tuple[int, float]]:
    steps: list[tuple[int, float]] = []
    for match in re.finditer(r"(\d{3,4})\s*C", window or ""):
        temperature = int(match.group(1))
        nearby = (window or "")[match.end() : match.end() + 32]
        time_match = re.search(r"(?:/|for|near|about|\s)(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*h", nearby, flags=re.IGNORECASE)
        hold_h = float(time_match.group(2) or time_match.group(1)) if time_match else 1.0
        steps.append((temperature, hold_h))
    return steps


def build_thermal_cycle_rows(ht_class: str, window: str) -> pd.DataFrame:
    steps = _temperature_time_steps(window)
    if not steps:
        return pd.DataFrame(columns=["ht_class", "elapsed_h", "temperature_C", "stage"])

    rows: list[dict[str, object]] = [{"ht_class": ht_class, "elapsed_h": 0.0, "temperature_C": 25, "stage": "start"}]
    elapsed_h = 0.0
    for idx, (temperature, hold_h) in enumerate(steps, start=1):
        ramp_h = max(abs(temperature - float(rows[-1]["temperature_C"])) / 600.0, 0.25)
        elapsed_h += ramp_h
        rows.append({"ht_class": ht_class, "elapsed_h": round(elapsed_h, 2), "temperature_C": temperature, "stage": f"ramp {idx}"})
        elapsed_h += hold_h
        rows.append({"ht_class": ht_class, "elapsed_h": round(elapsed_h, 2), "temperature_C": temperature, "stage": f"hold {idx}"})

    elapsed_h += max(abs(float(rows[-1]["temperature_C"]) - 25.0) / 450.0, 0.5)
    rows.append({"ht_class": ht_class, "elapsed_h": round(elapsed_h, 2), "temperature_C": 25, "stage": "cool"})
    return pd.DataFrame(rows)


def _normalise_to_100(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([50.0] * len(series), index=series.index)
    low = float(numeric.min())
    high = float(numeric.max())
    if high == low:
        return pd.Series([75.0] * len(series), index=series.index)
    return ((numeric - low) / (high - low) * 100.0).fillna(50.0)


def build_route_radar_rows(routes: pd.DataFrame) -> pd.DataFrame:
    if routes.empty:
        return pd.DataFrame(columns=["ht_class", "axis", "value"])

    scored = routes.copy()
    axis_sources = {
        "UTS": "predicted_UTS_MPa",
        "Yield strength": "predicted_YS_MPa",
        "Elongation": "predicted_elongation_pct",
        "Evidence support": "evidence_count_seed",
        "Recommendation index": "ml_assisted_score",
    }
    rows: list[dict[str, object]] = []
    for axis, column in axis_sources.items():
        values = _normalise_to_100(scored[column]) if column in scored.columns else pd.Series([50.0] * len(scored), index=scored.index)
        for idx, value in values.items():
            rows.append({"ht_class": scored.loc[idx, "ht_class"], "axis": axis, "value": round(float(value), 2)})
    return pd.DataFrame(rows)


def build_recommendation_contribution_rows(row: dict | pd.Series) -> pd.DataFrame:
    item = dict(row)
    adjusted_score = float(item.get("adjusted_score", 0.0) or 0.0)
    property_index = float(item.get("ml_property_index", 0.5) or 0.5)
    final_score = float(item.get("ml_assisted_score", 0.7 * adjusted_score + 0.3 * property_index) or 0.0)
    return pd.DataFrame(
        [
            {"term": "Evidence and feasibility", "measure": "relative", "value": round(0.7 * adjusted_score, 4)},
            {"term": "Calibrated property contribution", "measure": "relative", "value": round(0.3 * property_index, 4)},
            {"term": "Recommendation index", "measure": "total", "value": round(final_score, 4)},
        ]
    )
