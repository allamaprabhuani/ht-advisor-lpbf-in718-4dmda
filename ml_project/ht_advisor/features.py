from __future__ import annotations

import math
from typing import Any


def larson_miller_dose(temperature_C: float | int | str | None, time_h: float | int | str | None, C: float = 20.0) -> float:
    if temperature_C in ("", None) or time_h in ("", None):
        return 0.0
    t = max(float(time_h), 1e-6)
    temperature_K = float(temperature_C) + 273.15
    return temperature_K * (math.log10(t) + C)


def recipe_features(ht_class: str, steps: list[dict[str, Any]]) -> dict[str, float | int | str]:
    temps = [float(s["temperature_C"]) for s in steps if s.get("temperature_C") not in ("", None)]
    times = [float(s["time_h"]) for s in steps if s.get("time_h") not in ("", None)]
    ageing_steps = [s for s in steps if str(s.get("step_type", "")).lower() == "ageing"]
    solution_steps = [s for s in steps if str(s.get("step_type", "")).lower() in {"solution", "homogenisation", "homogenization"}]
    hip_steps = [s for s in steps if str(s.get("step_type", "")).lower() == "hip"]
    return {
        "ht_class": ht_class,
        "max_temperature_C": max(temps) if temps else 0.0,
        "total_time_h": sum(times),
        "number_of_thermal_steps": len(steps),
        "has_HIP": int(bool(hip_steps) or "HIP" in ht_class),
        "has_solution": int(bool(solution_steps) or "ST" in ht_class or "HA" in ht_class),
        "has_double_ageing": int(len(ageing_steps) >= 2 or "DA" in ht_class),
        "solution_thermal_dose": sum(larson_miller_dose(s.get("temperature_C"), s.get("time_h")) for s in solution_steps),
        "ageing_thermal_dose": sum(larson_miller_dose(s.get("temperature_C"), s.get("time_h")) for s in ageing_steps),
        "hip_thermal_dose": sum(larson_miller_dose(s.get("temperature_C"), s.get("time_h")) for s in hip_steps),
    }

