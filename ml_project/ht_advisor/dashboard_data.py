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
