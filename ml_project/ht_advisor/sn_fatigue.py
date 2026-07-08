from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


SN_POINT_COLUMNS = [
    "sn_point_id",
    "source_type",
    "target_id",
    "source_id",
    "source_pdf",
    "source_page",
    "figure_id",
    "curve_id",
    "condition_id",
    "point_index",
    "alloy",
    "test_type",
    "stress_metric_digitised",
    "stress_metric_type",
    "axis_scale_x",
    "axis_scale_y",
    "runout_encoding",
    "stress_amplitude_MPa",
    "max_stress_MPa",
    "cycles_to_failure",
    "log10_cycles_to_failure",
    "runout_flag",
    "stress_ratio_R",
    "test_temperature_C",
    "build_orientation",
    "surface_condition",
    "heat_treatment_class",
    "data_origin",
    "data_status",
    "review_status",
    "notes",
]


@dataclass(frozen=True)
class PlotCalibration:
    image_path: str
    x0_px: float
    x1_px: float
    y0_px: float
    y1_px: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_log: bool = True
    y_log: bool = False

    def to_data(self, x_px: float, y_px: float) -> tuple[float, float]:
        if self.x_log:
            log_x = math.log10(self.x_min) + (x_px - self.x0_px) / (self.x1_px - self.x0_px) * (
                math.log10(self.x_max) - math.log10(self.x_min)
            )
            x_value = 10**log_x
        else:
            x_value = self.x_min + (x_px - self.x0_px) / (self.x1_px - self.x0_px) * (self.x_max - self.x_min)
        if self.y_log:
            log_y = math.log10(self.y_min) + (self.y1_px - y_px) / (self.y1_px - self.y0_px) * (
                math.log10(self.y_max) - math.log10(self.y_min)
            )
            y_value = 10**log_y
        else:
            y_value = self.y_min + (self.y1_px - y_px) / (self.y1_px - self.y0_px) * (self.y_max - self.y_min)
        return x_value, y_value


def _marker_components(
    image_path: Path,
    crop: tuple[int, int, int, int],
    mask: np.ndarray,
    calibration: PlotCalibration,
    condition: str,
    legend_exclusion: tuple[int, int] | None = None,
) -> list[dict[str, object]]:
    x0, y0, x1, y1 = crop
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    points: list[dict[str, object]] = []
    for idx in range(1, n_labels):
        x, y, width, height, area = stats[idx]
        if not (12 <= area <= 260 and 5 <= width <= 24 and 5 <= height <= 24):
            continue
        aspect_ratio = max(width, height) / max(1, min(width, height))
        if aspect_ratio >= 2.2:
            continue
        cx, cy = centroids[idx]
        gx = float(cx + x0)
        gy = float(cy + y0)
        if legend_exclusion is not None and gx > legend_exclusion[0] and gy < legend_exclusion[1]:
            continue
        cycles, stress = calibration.to_data(gx, gy)
        points.append(
            {
                "heat_treatment_class": condition,
                "cycles_to_failure": cycles,
                "stress_amplitude_MPa": stress,
                "x_pixel": gx,
                "y_pixel": gy,
                "marker_area_px": int(area),
                "digitised_image": str(image_path),
            }
        )
    return sorted(points, key=lambda item: float(item["cycles_to_failure"]))


def _extract_new10_points() -> list[dict[str, object]]:
    image_path = ROOT / "ml_project" / "figures" / "sn_priority_rendered" / "NEW10_scan-11.png"
    if not image_path.exists():
        return []
    image = cv2.imread(str(image_path))
    crop = (172, 139, 694, 559)
    x0, y0, x1, y1 = crop
    plot = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
    masks = {
        "HT-1": cv2.inRange(hsv, np.array([0, 60, 80]), np.array([10, 255, 255]))
        | cv2.inRange(hsv, np.array([170, 60, 80]), np.array([179, 255, 255])),
        "HT-2": cv2.inRange(hsv, np.array([35, 45, 60]), np.array([90, 255, 255])),
        "HT-3": cv2.inRange(hsv, np.array([95, 45, 60]), np.array([125, 255, 255])),
        "As-SLM": cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 90, 115])),
    }
    calibration = PlotCalibration(
        image_path=str(image_path),
        x0_px=172,
        x1_px=694,
        y0_px=139,
        y1_px=559,
        x_min=1e5,
        x_max=1e9,
        y_min=150,
        y_max=900,
    )
    rows: list[dict[str, object]] = []
    for condition, mask in masks.items():
        rows.extend(
            _marker_components(
                image_path=image_path,
                crop=crop,
                mask=mask,
                calibration=calibration,
                condition=condition,
                legend_exclusion=(330, 280),
            )
        )
    return rows


def _extract_new17_points() -> list[dict[str, object]]:
    image_path = ROOT / "ml_project" / "figures" / "sn_priority_rendered" / "NEW17_p090-090.png"
    if not image_path.exists():
        return []
    image = cv2.imread(str(image_path))
    crop = (380, 300, 1530, 830)
    x0, y0, x1, y1 = crop
    plot = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
    purple_mask = cv2.inRange(hsv, np.array([115, 40, 40]), np.array([165, 255, 255]))
    calibration = PlotCalibration(
        image_path=str(image_path),
        x0_px=414,
        x1_px=1510,
        y0_px=326,
        y1_px=809,
        x_min=1e4,
        x_max=1e7,
        y_min=0,
        y_max=900,
    )
    rows = _marker_components(
        image_path=image_path,
        crop=crop,
        mask=purple_mask,
        calibration=calibration,
        condition="HIP+SA",
    )
    for row in rows:
        stress = float(row["stress_amplitude_MPa"])
        cycles = float(row["cycles_to_failure"])
        row["runout_flag"] = cycles > 9.5e5 and any(abs(stress - value) < 35 for value in [246, 422, 528])
    return rows


def _standard_point_row(
    point: dict[str, object],
    source_id: str,
    source_pdf: str,
    source_page: int,
    figure_id: str,
    condition_id: str,
    stress_ratio_R: str,
    point_index: int,
    notes: str,
) -> dict[str, object]:
    cycles = float(point["cycles_to_failure"])
    stress = float(point["stress_amplitude_MPa"])
    origin_path = Path(str(point["digitised_image"]))
    try:
        origin_rendered = str(origin_path.relative_to(ROOT))
    except ValueError:
        origin_rendered = str(origin_path)
    safe_figure_id = figure_id.replace(".", "").replace(" ", "_")
    row = {column: "" for column in SN_POINT_COLUMNS}
    row.update(
        {
            "sn_point_id": f"{condition_id}-{point_index:04d}",
            "source_type": "literature_digitised_marker",
            "target_id": f"SN_{source_id}_{safe_figure_id}",
            "source_id": source_id,
            "source_pdf": source_pdf,
            "source_page": source_page,
            "figure_id": figure_id,
            "curve_id": str(point["heat_treatment_class"]),
            "condition_id": condition_id,
            "point_index": point_index,
            "alloy": "Inconel 718",
            "test_type": "fatigue",
            "stress_metric_digitised": "stress amplitude",
            "stress_metric_type": "stress_amplitude",
            "axis_scale_x": "log",
            "axis_scale_y": "linear",
            "runout_encoding": "arrow marker retained as right-censored runout" if point.get("runout_flag") else "none observed",
            "stress_amplitude_MPa": round(stress, 2),
            "cycles_to_failure": round(cycles, 0),
            "log10_cycles_to_failure": round(math.log10(cycles), 5),
            "runout_flag": bool(point.get("runout_flag", False)),
            "stress_ratio_R": stress_ratio_R,
            "test_temperature_C": 25,
            "build_orientation": "not reported",
            "surface_condition": "not reported",
            "heat_treatment_class": str(point["heat_treatment_class"]),
            "data_origin": origin_rendered,
            "data_status": "marker_digitised_visual_reviewed",
            "review_status": "reviewed",
            "notes": notes,
        }
    )
    return row


def build_reviewed_sn_points() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, point in enumerate(_extract_new10_points(), start=1):
        condition = str(point["heat_treatment_class"]).replace("-", "").replace("+", "_")
        rows.append(
            _standard_point_row(
                point=point,
                source_id="NEW10",
                source_pdf="NEW10_Heat_treatment_for_selective_laser_melting_of_Inconel_718_alloy_with_simultaneously_enhanc.pdf",
                source_page=11,
                figure_id="Fig. 14",
                condition_id=f"NEW10_{condition}_Rminus1_RT",
                stress_ratio_R="-1",
                point_index=index,
                notes=(
                    "Marker-level point extracted from Fig. 14. Published caption reports R = -1 at room temperature; "
                    "fitted curve in the figure was not digitised as a point."
                ),
            )
        )
    new17_points = _extract_new17_points()
    for index, point in enumerate(new17_points, start=1):
        rows.append(
            _standard_point_row(
                point=point,
                source_id="NEW17",
                source_pdf="NEW17_MARINO_THESIS_2022.pdf",
                source_page=90,
                figure_id="Fig. 4.15",
                condition_id="NEW17_HIP_SA_Rnotreported_RT",
                stress_ratio_R="not reported",
                point_index=index,
                notes=(
                    "Marker-level point extracted from Fig. 4.15. The thesis reports ASTM E466 force-controlled axial fatigue; "
                    "stress ratio is not reported in the extracted text. Arrow-marked markers at 1e6 cycles are retained as runouts."
                ),
            )
        )
    return pd.DataFrame(rows, columns=SN_POINT_COLUMNS)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def fit_basquin_models(points: pd.DataFrame, minimum_failures: int = 3) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for condition_id, subset in points.groupby("condition_id"):
        data = subset.copy()
        data["cycles_to_failure"] = pd.to_numeric(data["cycles_to_failure"], errors="coerce")
        data["stress_amplitude_MPa"] = pd.to_numeric(data["stress_amplitude_MPa"], errors="coerce")
        data["runout_flag"] = data["runout_flag"].map(_as_bool)
        failures = data[(~data["runout_flag"]) & data["cycles_to_failure"].gt(0) & data["stress_amplitude_MPa"].gt(0)]
        runouts = data[data["runout_flag"]]
        if len(failures) < minimum_failures:
            continue
        x = np.log10(2.0 * failures["cycles_to_failure"].to_numpy(dtype=float))
        y = np.log10(failures["stress_amplitude_MPa"].to_numpy(dtype=float))
        slope, intercept = np.polyfit(x, y, deg=1)
        if slope >= 0:
            continue
        censor_shift = 0.0
        runout_margins: list[float] = []
        if not runouts.empty:
            runout_x = np.log10(2.0 * runouts["cycles_to_failure"].to_numpy(dtype=float))
            runout_y = np.log10(runouts["stress_amplitude_MPa"].to_numpy(dtype=float))
            initial_margins = intercept + slope * runout_x - runout_y
            censor_shift = max(0.0, float(-initial_margins.min()) + 0.001)
            intercept += censor_shift
            runout_margins = (intercept + slope * runout_x - runout_y).tolist()
        fitted = intercept + slope * x
        residuals = y - fitted
        rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0
        satisfied_runouts = sum(1 for margin in runout_margins if margin >= 0.0)
        min_runout_margin = min(runout_margins) if runout_margins else float("nan")
        representative = data.iloc[0]
        rows.append(
            {
                "condition_id": condition_id,
                "source_id": representative.get("source_id", ""),
                "source_pdf": representative.get("source_pdf", ""),
                "source_page": representative.get("source_page", ""),
                "figure_id": representative.get("figure_id", ""),
                "heat_treatment_class": representative.get("heat_treatment_class", ""),
                "stress_ratio_R": str(representative.get("stress_ratio_R", "")),
                "surface_condition": representative.get("surface_condition", ""),
                "build_orientation": representative.get("build_orientation", ""),
                "n_points": int(len(data)),
                "n_failures": int(len(failures)),
                "n_runouts": int(len(runouts)),
                "cycle_min": float(failures["cycles_to_failure"].min()),
                "cycle_max": float(failures["cycles_to_failure"].max()),
                "stress_min_MPa": float(failures["stress_amplitude_MPa"].min()),
                "stress_max_MPa": float(failures["stress_amplitude_MPa"].max()),
                "basquin_log10_sigma_f": round(float(intercept), 6),
                "basquin_slope_b": round(float(slope), 6),
                "rmse_log10_stress": round(max(rmse, 0.03), 6),
                "censoring_method": (
                    "failure-point Basquin fit with right-censored runout lower-bound adjustment"
                    if len(runouts)
                    else "failure-point Basquin fit; no right-censored runouts in this condition"
                ),
                "censor_intercept_shift_log10_stress": round(float(censor_shift), 6),
                "n_censored_constraints_satisfied": int(satisfied_runouts),
                "runout_margin_log10_stress": round(float(min_runout_margin), 6) if runout_margins else "",
                "model_status": "trained_condition_specific_censored_basquin_screening",
                "model_boundary": (
                    "Condition-specific Basquin regression fitted to reviewed marker points. Failure points set the least-squares "
                    "slope and right-censored runouts are used as lower-bound constraints on the fitted stress-life curve. "
                    "Do not pool across stress ratio, surface condition, or heat-treatment state."
                ),
            }
        )
    return pd.DataFrame(rows)


def make_prediction_grid(model_summary: pd.DataFrame, points_per_condition: int = 60) -> pd.DataFrame:
    if model_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, model in model_summary.iterrows():
        cycles = np.logspace(math.log10(float(model["cycle_min"])), math.log10(float(model["cycle_max"])), points_per_condition)
        intercept = float(model["basquin_log10_sigma_f"])
        slope = float(model["basquin_slope_b"])
        rmse = float(model["rmse_log10_stress"])
        for cycle in cycles:
            log_stress = intercept + slope * math.log10(2.0 * cycle)
            stress = 10**log_stress
            rows.append(
                {
                    "condition_id": model["condition_id"],
                    "source_id": model["source_id"],
                    "heat_treatment_class": model["heat_treatment_class"],
                    "stress_ratio_R": model["stress_ratio_R"],
                    "cycles_to_failure": round(float(cycle), 0),
                    "stress_amplitude_MPa": round(float(stress), 2),
                    "stress_lower_MPa": round(float(10 ** (log_stress - 1.96 * rmse)), 2),
                    "stress_upper_MPa": round(float(10 ** (log_stress + 1.96 * rmse)), 2),
                    "curve_style": "dashed_literature_marker_fit",
                    "model_boundary": model["model_boundary"],
                }
            )
    return pd.DataFrame(rows)


def _stress_components(stress_amplitude_MPa: float, stress_ratio_R: float) -> tuple[float, float, float]:
    if stress_ratio_R >= 1.0:
        raise ValueError("stress_ratio_R must be less than 1.0")
    sigma_max = 2.0 * float(stress_amplitude_MPa) / (1.0 - float(stress_ratio_R))
    sigma_min = float(stress_ratio_R) * sigma_max
    sigma_mean = 0.5 * (sigma_max + sigma_min)
    return sigma_max, sigma_min, sigma_mean


def goodman_equivalent_reversed_amplitude(
    stress_amplitude_MPa: float,
    stress_ratio_R: float,
    uts_MPa: float,
) -> float:
    """Convert a stress amplitude at arbitrary R to an equivalent R = -1 amplitude.

    This is a Goodman mean-stress screening correction, not an additional
    fatigue calibration. It is used only to compare proposed tests against
    reviewed fully reversed literature S-N curves.
    """
    _, _, sigma_mean = _stress_components(float(stress_amplitude_MPa), float(stress_ratio_R))
    denominator = 1.0 - sigma_mean / float(uts_MPa)
    if denominator <= 0.0:
        return float("inf")
    return float(stress_amplitude_MPa) / denominator


def stress_amplitude_for_goodman_target(
    equivalent_R_minus_1_amplitude_MPa: float,
    stress_ratio_R: float,
    uts_MPa: float,
) -> float:
    """Translate an R = -1 stress amplitude to another R using Goodman correction."""
    if stress_ratio_R >= 1.0:
        raise ValueError("stress_ratio_R must be less than 1.0")
    mean_factor = (1.0 + float(stress_ratio_R)) / (1.0 - float(stress_ratio_R))
    denominator = 1.0 + float(equivalent_R_minus_1_amplitude_MPa) * mean_factor / float(uts_MPa)
    if denominator <= 0.0:
        return float("nan")
    return float(equivalent_R_minus_1_amplitude_MPa) / denominator


def _basquin_stress_for_life(model: pd.Series, cycles_to_failure: float) -> float:
    intercept = float(model["basquin_log10_sigma_f"])
    slope = float(model["basquin_slope_b"])
    log_stress = intercept + slope * math.log10(2.0 * float(cycles_to_failure))
    return 10**log_stress


def _basquin_life_for_stress(model: pd.Series, stress_amplitude_MPa: float) -> float:
    intercept = float(model["basquin_log10_sigma_f"])
    slope = float(model["basquin_slope_b"])
    log_reversals = (math.log10(float(stress_amplitude_MPa)) - intercept) / slope
    return (10**log_reversals) / 2.0


def build_stress_ratio_screening_table(
    model_summary: pd.DataFrame,
    stress_ratios: list[float] | tuple[float, ...] = (-1.0, 0.0, 0.1),
    target_cycles: list[int] | tuple[int, ...] = (100000, 300000, 1000000, 3000000, 10000000),
    uts_MPa: float = 1350.0,
    source_curve_stress_ratio_R: str = "-1",
) -> pd.DataFrame:
    """Build target-life stress amplitudes for multiple R values from reviewed R = -1 curves.

    The returned rows are mean-stress-corrected screening values. They do not
    create new R-specific training data and should not be reported as local
    fatigue allowables.
    """
    if model_summary.empty:
        return pd.DataFrame()
    source_models = model_summary[model_summary["stress_ratio_R"].astype(str).eq(str(source_curve_stress_ratio_R))]
    rows: list[dict[str, object]] = []
    for _, model in source_models.iterrows():
        for life in target_cycles:
            equivalent_amplitude = _basquin_stress_for_life(model, float(life))
            for ratio in stress_ratios:
                screening_amplitude = stress_amplitude_for_goodman_target(equivalent_amplitude, float(ratio), float(uts_MPa))
                sigma_max, sigma_min, sigma_mean = _stress_components(screening_amplitude, float(ratio))
                translated_life = _basquin_life_for_stress(
                    model,
                    goodman_equivalent_reversed_amplitude(screening_amplitude, float(ratio), float(uts_MPa)),
                )
                rows.append(
                    {
                        "condition_id": model["condition_id"],
                        "source_id": model.get("source_id", ""),
                        "source_curve_stress_ratio_R": str(model["stress_ratio_R"]),
                        "heat_treatment_class": model.get("heat_treatment_class", ""),
                        "target_cycles": int(life),
                        "screening_stress_ratio_R": round(float(ratio), 3),
                        "screening_stress_amplitude_MPa": round(float(screening_amplitude), 1),
                        "sigma_max_MPa": round(float(sigma_max), 1),
                        "sigma_min_MPa": round(float(sigma_min), 1),
                        "sigma_mean_MPa": round(float(sigma_mean), 1),
                        "goodman_equivalent_R_minus_1_MPa": round(float(equivalent_amplitude), 1),
                        "translated_life_cycles": round(float(translated_life), 0),
                        "uts_assumption_MPa": round(float(uts_MPa), 1),
                        "screening_boundary": (
                            "Goodman mean-stress translation from reviewed R = -1 Basquin literature curves; "
                            "not a separately trained stress-ratio-specific fatigue-life predictor."
                        ),
                    }
                )
    return pd.DataFrame(rows)
