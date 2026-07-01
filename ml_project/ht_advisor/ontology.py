from __future__ import annotations

import re

HT_CLASSES = ["AB", "SR", "DA", "ST_DA", "HIP_DA", "HIP_ST_DA", "HA_ST_DA", "CUSTOM_ST_DA"]


def normalize_ht_class(value: str) -> str:
    x = (value or "").strip().upper().replace(" ", "").replace("-", "_").replace("+", "_")
    aliases = {
        "HP_ST_DA": "HIP_ST_DA",
        "HIP_ST_DA": "HIP_ST_DA",
        "ST_DA": "ST_DA",
        "HIP_DA": "HIP_DA",
        "HP_DA": "HIP_DA",
        "HA_ST_DA": "HA_ST_DA",
        "DA": "DA",
        "SR": "SR",
        "AB": "AB",
    }
    return aliases.get(x, value or "")


def classify_ht_text(text: str) -> str:
    x = (text or "").lower()
    has_hip = "hip" in x or "hot isostatic" in x
    has_homogen = "homogen" in x
    has_solution = "solution" in x or "anneal" in x or re.search(r"\b9[5-9]\d\s*°?\s*c\b", x) or re.search(r"\b10\d\d\s*°?\s*c\b", x)
    has_age = "age" in x or "aging" in x or "ageing" in x or "720" in x or "620" in x or "621" in x or "718" in x
    has_stress_relief = "stress relief" in x or "stress-relief" in x or "stress relieved" in x
    if has_hip and has_solution and has_age:
        return "HIP_ST_DA"
    if has_homogen and has_solution and has_age:
        return "HA_ST_DA"
    if has_hip and has_age:
        return "HIP_DA"
    if has_solution and has_age:
        return "ST_DA"
    if has_age:
        return "DA"
    if has_stress_relief:
        return "SR"
    return "AB"


def is_surface_treatment(text: str) -> bool:
    x = (text or "").lower()
    return any(term in x for term in ["shot peen", "lpb", "polish", "machin", "surface", "oxid"])
