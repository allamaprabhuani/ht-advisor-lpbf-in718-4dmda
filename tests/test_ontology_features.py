from ml_project.ht_advisor.features import larson_miller_dose, recipe_features
from ml_project.ht_advisor.ontology import classify_ht_text, normalize_ht_class


def test_classify_ht_text_st_da():
    text = "980 C for 1 h + 720 C for 8 h furnace cool to 620 C for 8 h"
    assert classify_ht_text(text) == "ST_DA"


def test_classify_ht_text_hip_st_da():
    text = "HIP at 1163 C and 100 MPa followed by solution anneal and dual ageing"
    assert classify_ht_text(text) == "HIP_ST_DA"


def test_normalize_ht_class_from_excel_style():
    assert normalize_ht_class("HIP+ST+DA") == "HIP_ST_DA"
    assert normalize_ht_class("ST+DA") == "ST_DA"


def test_larson_miller_dose_increases_with_temperature():
    low = larson_miller_dose(720, 8)
    high = larson_miller_dose(980, 8)
    assert high > low


def test_recipe_features_has_flags():
    steps = [
        {"step_type": "solution", "temperature_C": 980, "time_h": 1},
        {"step_type": "ageing", "temperature_C": 720, "time_h": 8},
        {"step_type": "ageing", "temperature_C": 620, "time_h": 8},
    ]
    feats = recipe_features("ST_DA", steps)
    assert feats["has_solution"] == 1
    assert feats["has_double_ageing"] == 1
    assert feats["max_temperature_C"] == 980
