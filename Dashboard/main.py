"""FastAPI dashboard for the Road RSL Prediction project.

Supervised dual time-series forecasting:
1. Model 1 (XGBoost Regressor): Surface roughness (IRI / MRI) deterioration.
2. Model 2 (XGBoost Regressor): Structural layer fatigue (Surface Curvature Index SCI / BDI) deterioration.
Outputs a synchronized 50/50 Road Health Index (RHI) across both historical snapshot and present day (2026).
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from xgboost import DMatrix


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"
SECTION_COLUMNS = ["SHRP_ID", "STATE_CODE", "CONSTRUCTION_NO"]

IRI_FEATURES = [
    "MRI", "AADTT_ALL_TRUCKS_TREND", "ANNUAL_TRUCK_VOLUME_TREND",
    "ANNUAL_ESAL_TREND", "CUMULATIVE_ESAL", "YEAR",
    "MEAN_ANN_TEMP_AVG", "FREEZE_INDEX_YR", "FREEZE_THAW_YR",
]

SCI_FEATURES = [
    "SCI", "BDI", "DROP_LOAD", "DROP_HEIGHT",
    "PAVEMENT_FAMILY_ENC", "LANE_NO_ENC",
    "AADTT_ALL_TRUCKS_TREND", "ANNUAL_TRUCK_VOLUME_TREND", "ANNUAL_ESAL_TREND",
    "CUMULATIVE_ESAL", "YEAR", "YEARS_SINCE_LAST_REPAIR",
    "MEAN_ANN_TEMP_AVG", "FREEZE_INDEX_YR", "FREEZE_THAW_YR",
]

IRI_FAILURE_THRESHOLD = 2.5
SCI_FAILURE_THRESHOLD = 200.0


class PredictionInput(BaseModel):
    mri: float = Field(..., ge=0, le=10)
    aadtt: float = Field(..., ge=0, le=50_000)
    annual_truck_volume: float = Field(..., ge=0, le=20_000_000)
    annual_esal: float = Field(..., ge=0, le=50_000_000)
    cumulative_esal: float = Field(..., ge=0, le=500_000_000)
    year: int = Field(..., ge=1980, le=2030)
    mean_ann_temp_avg: float = Field(..., ge=-100, le=100)
    freeze_index_yr: float = Field(..., ge=0, le=100_000)
    freeze_thaw_yr: float = Field(..., ge=0, le=100_000)
    fwd_available: bool = True
    deflections: list[float] | None = None
    drop_load: float | None = Field(default=None, ge=0, le=2_000)
    drop_height: int | None = Field(default=None, ge=1, le=4)
    pavement_family: str | None = None
    lane_no: str | None = None

    @field_validator("year")
    @classmethod
    def validate_year(cls, value: int) -> int:
        if value < 1980 or value > 2030:
            raise ValueError("Measurement year must be between 1980 and 2030.")
        return value

    @field_validator("deflections")
    @classmethod
    def validate_deflections(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        if len(value) != 7:
            raise ValueError("Provide exactly seven FWD deflection values (D1 through D7).")
        if any(item < 0 or item > 2_000 for item in value):
            raise ValueError("Each FWD deflection must be between 0 and 2,000 microns.")
        return value


app = FastAPI(title="Road Health Index Dashboard", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("DASHBOARD_ALLOWED_ORIGIN", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_id(series: pd.Series) -> pd.Series:
    """Zero-pad a SHRP_ID series to 4 digits, stripping float suffixes."""
    return series.astype(str).str.replace(".0", "", regex=False).str.zfill(4)


def condition(score: float) -> str:
    """Map a 0-100 RHI score to an FHWA-aligned pavement condition label."""
    if score >= 75.0:
        return "Good"
    if score >= 50.0:
        return "Fair"
    return "Poor"


def recommendation(score: float) -> str:
    """Return a prescriptive maintenance policy string based on the RHI score."""
    if score >= 75.0:
        return "Routine preventive preservation; both surface roughness and asphalt structural integrity are in good condition."
    if score >= 50.0:
        return "Plan targeted resurfacing and investigate structural micro-fatigue before deeper base deterioration accelerates."
    return "Prioritize structural rehabilitation and deep milling/paving to restore load-bearing capacity and surface ride quality."


def score_iri(iri_val: float) -> float:
    """Convert IRI (m/km) to a 0-100 surface health score.

    Formula: ((2.5 - IRI) / 2.5) * 100, clipped to [0, 100].
    Critical failure at IRI >= 2.5 m/km (FHWA Pavement Design Guide).
    """
    return float(np.clip(((IRI_FAILURE_THRESHOLD - iri_val) / IRI_FAILURE_THRESHOLD) * 100.0, 0.0, 100.0))


def score_sci(sci_val: float) -> float:
    """Convert temperature-normalized SCI (µm) to a 0-100 structural health score.

    Formula: ((200 - SCI) / 200) * 100, clipped to [0, 100].
    Critical failure at SCI >= 200 µm (AASHTO Pavement Design Guide fatigue threshold).
    """
    return float(np.clip(((SCI_FAILURE_THRESHOLD - sci_val) / SCI_FAILURE_THRESHOLD) * 100.0, 0.0, 100.0))


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    """Load and cache all ML model binaries and calibrated fallback rates from disk.

    Returns a dictionary containing the trained XGBoost regressors, LabelEncoders,
    and physics-calibrated annual deterioration fallback rates for both IRI and SCI.
    Raises HTTP 503 if any required model artifact is missing.
    """
    required = {
        "iri_model": "iri_prediction_model.pkl",
        "sci_model": "sci_prediction_model.pkl",
        "pavement_encoder": "sci_le_pav.pkl",
        "lane_encoder": "sci_le_lane.pkl",
    }
    missing = [filename for filename in required.values() if not (MODEL_DIR / filename).exists()]
    if missing:
        raise HTTPException(503, f"Missing model artifacts: {', '.join(missing)}. Run model training first.")
    artifacts = {name: joblib.load(MODEL_DIR / filename) for name, filename in required.items()}

    # Fallback degradation rates
    det_rate_file = MODEL_DIR / "deterioration_rate.txt"
    if det_rate_file.exists():
        try:
            with open(det_rate_file, "r") as f:
                artifacts["iri_deterioration_rate"] = float(f.read().strip())
        except Exception:
            artifacts["iri_deterioration_rate"] = 0.04
    else:
        artifacts["iri_deterioration_rate"] = 0.04

    sci_det_file = MODEL_DIR / "sci_deterioration_rate.txt"
    if sci_det_file.exists():
        try:
            with open(sci_det_file, "r") as f:
                artifacts["sci_deterioration_rate"] = float(f.read().strip())
        except Exception:
            artifacts["sci_deterioration_rate"] = 4.2
    else:
        artifacts["sci_deterioration_rate"] = 4.2

    return artifacts


@lru_cache(maxsize=1)
def load_network_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return prepared IRI records and FWD records from disk cache or source workbooks."""
    cache_path = DATA_DIR / "processed_network_cache.pkl"
    if cache_path.exists():
        try:
            iri_records, fwd_records = joblib.load(cache_path)
            return iri_records, fwd_records
        except Exception:
            pass

    try:
        iri = pd.read_excel(DATA_DIR / "MON_HSS_PROFILE_SECTION.xlsx")
        trf1 = pd.read_excel(DATA_DIR / "TRF_TREND_1.xlsx")
        trf2 = pd.read_excel(DATA_DIR / "TRF_TREND.xlsx")
        climate = pd.read_excel(DATA_DIR / "CLM_VWS_TEMP_ANNUAL.xlsx")
        fwd = pd.read_excel(DATA_DIR / "MON_DEFL_DROP_DATA.xlsx")
        exp = pd.read_excel(DATA_DIR / "EXPERIMENT_SECTION.xlsx")
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Required source data is unavailable: {exc.filename}") from exc

    for frame in (iri, trf1, trf2, climate, fwd, exp):
        frame["SHRP_ID"] = normalize_id(frame["SHRP_ID"])
        frame["STATE_CODE"] = frame["STATE_CODE"].astype(str).str.replace(".0", "", regex=False)

    iri["YEAR"] = pd.to_datetime(iri["VISIT_DATE"]).dt.year
    iri_clean = iri.groupby([*SECTION_COLUMNS, "YEAR"])["MRI"].mean().reset_index()
    trf1["YEAR"] = trf1["YEAR"].astype(int)
    trf2["YEAR"] = trf2["YEAR"].astype(int)
    traffic = pd.merge(
        trf1[[*SECTION_COLUMNS, "YEAR", "AADTT_ALL_TRUCKS_TREND", "ANNUAL_TRUCK_VOLUME_TREND"]],
        trf2[[*SECTION_COLUMNS, "YEAR", "ANNUAL_ESAL_TREND"]],
        on=[*SECTION_COLUMNS, "YEAR"], how="outer",
    )
    iri_records = pd.merge(iri_clean, traffic, on=[*SECTION_COLUMNS, "YEAR"], how="inner")
    climate_columns = ["SHRP_ID", "STATE_CODE", "YEAR", "MEAN_ANN_TEMP_AVG", "FREEZE_INDEX_YR", "FREEZE_THAW_YR"]
    iri_records = pd.merge(iri_records, climate[climate_columns], on=["SHRP_ID", "STATE_CODE", "YEAR"], how="inner")
    iri_records = iri_records.sort_values([*SECTION_COLUMNS, "YEAR"])
    for column in ["ANNUAL_ESAL_TREND", "AADTT_ALL_TRUCKS_TREND", "ANNUAL_TRUCK_VOLUME_TREND"]:
        iri_records[column] = iri_records.groupby(SECTION_COLUMNS)[column].ffill()
    iri_records = iri_records.dropna().copy()
    iri_records["CUMULATIVE_ESAL"] = iri_records.groupby(SECTION_COLUMNS)["ANNUAL_ESAL_TREND"].cumsum()

    exp_clean = exp[[*SECTION_COLUMNS, "PAVEMENT_FAMILY"]].drop_duplicates()
    fwd_records = pd.merge(fwd, exp_clean, on=SECTION_COLUMNS, how="inner")
    required_fwd = [f"PEAK_DEFL_{i}" for i in range(1, 8)] + ["DROP_LOAD", "DROP_HEIGHT", "PAVEMENT_FAMILY", "LANE_NO"]
    fwd_records = fwd_records.dropna(subset=required_fwd).copy()

    try:
        joblib.dump((iri_records, fwd_records), cache_path)
    except Exception:
        pass

    return iri_records, fwd_records


def predict(payload: PredictionInput) -> dict[str, Any]:
    """Execute the full dual-timeline Road Health Index prediction pipeline.

    Steps:
    1. Compute historical snapshot RHI at survey year using measured IRI and
       AASHTO BELLS temperature-normalized SCI from raw FWD deflections.
    2. Run synchronized year-by-year AI simulation (Model 1: IRI, Model 2: Delta SCI)
       from survey year to present day (2026), applying the Virtual Maintenance Trigger
       (SCI reset to 40 µm at 150 µm threshold) and bounded decay [1.5, 8.0] µm/yr.
    3. Compute SHAP-based feature importance contributions from Model 1.
    4. Project 10-year planning horizon (2026-2036) with the same physics constraints.
    5. Return a structured payload: historical snapshot, present-day estimation,
       simulation path, projection, explanation cards, and top-level RHI score.
    """
    artifacts = load_artifacts()

    # --- 1. HISTORICAL SNAPSHOT AT SURVEY DATE (payload.year) ---
    hist_year = int(payload.year)
    hist_mri = float(payload.mri)
    hist_iri_score = score_iri(hist_mri)

    has_fwd = bool(payload.fwd_available and payload.deflections and len(payload.deflections) == 7)
    hist_sci: float | None = None
    hist_bdi: float | None = None
    hist_sci_score: float | None = None
    pav_enc = 0
    lane_enc = 0
    drop_load = float(payload.drop_load or 710.0)
    drop_height = int(payload.drop_height or 4)

    if has_fwd:
        d = payload.deflections
        # Mechanistic Indices: SCI (D1 - D2) and BDI (D2 - D3) with AASHTO BELLS temperature normalization
        t_pav = float(payload.mean_ann_temp_avg)
        bells_factor = 10.0 ** (-0.0079 * (20.0 - t_pav))
        d1_norm = d[0] * bells_factor
        d2_norm = d[1] * bells_factor
        hist_sci = float(d1_norm - d2_norm)
        hist_bdi = float(d[1] - d[2])
        hist_sci_score = score_sci(hist_sci)

        try:
            pav_str = payload.pavement_family or "ACUB"
            lane_str = payload.lane_no or "F1"
            pav_enc = int(artifacts["pavement_encoder"].transform([pav_str])[0])
            lane_enc = int(artifacts["lane_encoder"].transform([lane_str])[0])
        except Exception:
            pav_enc = 0
            lane_enc = 0

    # Historical Synchronized RHI (50% Surface + 50% Structural)
    if has_fwd and hist_sci_score is not None:
        hist_rhi = float((hist_iri_score * 0.50) + (hist_sci_score * 0.50))
        hist_fwd_health = condition(hist_sci_score)
    else:
        hist_rhi = hist_iri_score
        hist_fwd_health = "Not provided"

    hist_condition = condition(hist_rhi)

    # --- 2. CONCURRENT DUAL AI TIME-SERIES FAST-FORWARD TO PRESENT DAY (2026) ---
    target_present_year = 2026
    current_mri = hist_mri
    current_sci = hist_sci
    current_bdi = hist_bdi
    current_cum_esal = payload.cumulative_esal
    years_since_repair = 0

    simulation_path = [{
        "year": hist_year,
        "iri": round(hist_mri, 3),
        "iri_score": round(hist_iri_score, 1),
        "sci": round(hist_sci, 1) if hist_sci is not None else None,
        "sci_score": round(hist_sci_score, 1) if hist_sci_score is not None else None,
        "rhi": round(hist_rhi, 1),
    }]

    if hist_year < target_present_year:
        for yr in range(hist_year, target_present_year):
            # 1. Surface step forecast (Model 1)
            step_iri_input = pd.DataFrame([[
                current_mri, payload.aadtt, payload.annual_truck_volume,
                payload.annual_esal, current_cum_esal, yr,
                payload.mean_ann_temp_avg, payload.freeze_index_yr, payload.freeze_thaw_yr,
            ]], columns=IRI_FEATURES)
            raw_next_mri = float(artifacts["iri_model"].predict(step_iri_input)[0])
            next_mri = raw_next_mri if raw_next_mri > current_mri else current_mri + artifacts.get("iri_deterioration_rate", 0.04)
            current_mri = next_mri

            # 2. Structural step forecast (Model 2: Delta SCI + Virtual Maintenance Trigger)
            if has_fwd and current_sci is not None:
                step_sci_input = pd.DataFrame([[
                    current_sci, current_bdi, drop_load, drop_height,
                    pav_enc, lane_enc,
                    payload.aadtt, payload.annual_truck_volume, payload.annual_esal,
                    current_cum_esal, yr, years_since_repair,
                    payload.mean_ann_temp_avg, payload.freeze_index_yr, payload.freeze_thaw_yr,
                ]], columns=SCI_FEATURES)

                try:
                    predicted_delta_sci = float(artifacts["sci_model"].predict(step_sci_input)[0])
                    # Bounded decay rate: max(1.5, min(predicted_delta_sci, 8.0))
                    annual_degradation = max(1.5, min(predicted_delta_sci, 8.0))
                except Exception:
                    # Physics-backed calibrated default fallback
                    annual_degradation = float(artifacts.get("sci_deterioration_rate", 4.2))

                next_sci = current_sci + annual_degradation

                # Virtual Maintenance Trigger:
                # If simulated next_sci exceeds 150 µm (critical structural failure),
                # simulate physical overlay by resetting next_sci to baseline of 40 µm,
                # resetting years_since_repair to 0, and continuing the simulation loop naturally.
                if next_sci > 150.0:
                    next_sci = 40.0
                    years_since_repair = 0
                    if current_bdi is not None:
                        current_bdi = max(15.0, current_bdi * 0.5)
                else:
                    years_since_repair += 1

                if current_bdi is not None and next_sci != 40.0:
                    current_bdi = current_bdi * (next_sci / max(1.0, current_sci))
                current_sci = next_sci

            current_cum_esal += payload.annual_esal

            step_iri_s = score_iri(current_mri)
            step_sci_s = score_sci(current_sci) if (has_fwd and current_sci is not None) else None
            step_rhi = (step_iri_s * 0.50 + step_sci_s * 0.50) if step_sci_s is not None else step_iri_s

            simulation_path.append({
                "year": yr + 1,
                "iri": round(current_mri, 3),
                "iri_score": round(step_iri_s, 1),
                "sci": round(current_sci, 1) if current_sci is not None else None,
                "sci_score": round(step_sci_s, 1) if step_sci_s is not None else None,
                "rhi": round(step_rhi, 1),
            })
    else:
        current_mri = hist_mri
        current_sci = hist_sci

    # --- 3. PRESENT DAY (2026) ESTIMATION ---
    present_iri_score = score_iri(current_mri)

    if has_fwd and current_sci is not None:
        present_sci_score = score_sci(current_sci)
        present_rhi = float((present_iri_score * 0.50) + (present_sci_score * 0.50))
        structural_policy = "Synchronized 50/50 Dual AI Forecast (Surface + Structural Decay Projected to 2026)"
        present_fwd_health = condition(present_sci_score)
    else:
        present_sci_score = None
        present_rhi = present_iri_score
        structural_policy = "100% Surface AI Forecast (FWD deflection data not supplied)"
        present_fwd_health = "N/A"

    present_condition = condition(present_rhi)

    # --- 4. SHAP FEATURE EXPLANATION FOR 2026 ---
    explanation_input = pd.DataFrame([[
        current_mri, payload.aadtt, payload.annual_truck_volume, payload.annual_esal,
        current_cum_esal, target_present_year, payload.mean_ann_temp_avg,
        payload.freeze_index_yr, payload.freeze_thaw_yr,
    ]], columns=IRI_FEATURES)
    contributions = artifacts["iri_model"].get_booster().predict(
        DMatrix(explanation_input, feature_names=IRI_FEATURES), pred_contribs=True
    )[0][:-1]
    total_impact = sum(abs(float(value)) for value in contributions) or 1
    explanation = sorted([
        {"feature": feature.replace("_", " ").title(), "impact_percent": round(abs(float(value)) / total_impact * 100, 1),
         "direction": "accelerates deterioration" if value > 0 else "reduces deterioration rate"}
        for feature, value in zip(IRI_FEATURES, contributions)
    ], key=lambda item: item["impact_percent"], reverse=True)[:4]

    # --- 5. 10-YEAR HORIZON PROJECTION (2026 -> 2036) ---
    projected_iri = current_mri
    projected_sci = current_sci
    projected_bdi = current_bdi
    projected_esal = current_cum_esal
    projected_years_repair = years_since_repair
    projection = []

    for offset in range(1, 11):
        future_year = target_present_year + offset
        # Surface projection
        future_iri_input = pd.DataFrame([[
            projected_iri, payload.aadtt, payload.annual_truck_volume,
            payload.annual_esal, projected_esal + payload.annual_esal * offset, future_year,
            payload.mean_ann_temp_avg, payload.freeze_index_yr, payload.freeze_thaw_yr,
        ]], columns=IRI_FEATURES)
        raw_p_iri = float(artifacts["iri_model"].predict(future_iri_input)[0])
        projected_iri = raw_p_iri if raw_p_iri > projected_iri else projected_iri + artifacts.get("iri_deterioration_rate", 0.04)

        # Structural projection (Annualized Delta + Virtual Maintenance Trigger)
        if has_fwd and projected_sci is not None:
            future_sci_input = pd.DataFrame([[
                projected_sci, projected_bdi, drop_load, drop_height,
                pav_enc, lane_enc,
                payload.aadtt, payload.annual_truck_volume, payload.annual_esal,
                projected_esal + payload.annual_esal * offset, future_year, projected_years_repair,
                payload.mean_ann_temp_avg, payload.freeze_index_yr, payload.freeze_thaw_yr,
            ]], columns=SCI_FEATURES)

            try:
                predicted_p_delta = float(artifacts["sci_model"].predict(future_sci_input)[0])
                p_annual_deg = max(1.5, min(predicted_p_delta, 8.0))
            except Exception:
                p_annual_deg = float(artifacts.get("sci_deterioration_rate", 4.2))

            next_p_sci = projected_sci + p_annual_deg

            # Virtual Maintenance Trigger
            if next_p_sci > 150.0:
                next_p_sci = 40.0
                projected_years_repair = 0
                if projected_bdi is not None:
                    projected_bdi = max(15.0, projected_bdi * 0.5)
            else:
                projected_years_repair += 1

            if projected_bdi is not None and next_p_sci != 40.0:
                projected_bdi = projected_bdi * (next_p_sci / max(1.0, projected_sci))
            projected_sci = next_p_sci

        p_iri_score = score_iri(projected_iri)
        p_sci_score = score_sci(projected_sci) if (has_fwd and projected_sci is not None) else None
        p_rhi = (p_iri_score * 0.50 + p_sci_score * 0.50) if p_sci_score is not None else p_iri_score

        projection.append({
            "year": future_year,
            "iri": round(projected_iri, 3),
            "iri_score": round(p_iri_score, 1),
            "sci": round(projected_sci, 1) if projected_sci is not None else None,
            "sci_score": round(p_sci_score, 1) if p_sci_score is not None else None,
            "rhi": round(p_rhi, 1),
        })

    return {
        # Primary Present Day (2026) Results
        "rhi": round(present_rhi, 1),
        "condition": present_condition,
        "iri_score": round(present_iri_score, 1),
        "predicted_future_iri": round(current_mri, 3),
        "predicted_future_sci": round(current_sci, 1) if current_sci is not None else None,
        "fwd_score": round(present_sci_score, 1) if present_sci_score is not None else None,
        "fwd_health": present_fwd_health,
        "recommendation": recommendation(present_rhi),
        "explanation": explanation,
        "projection": projection,
        "simulation_path": simulation_path,

        # Dual Timeline Specific Objects
        "historical_snapshot": {
            "year": hist_year,
            "measured_iri": round(hist_mri, 3),
            "iri_score": round(hist_iri_score, 1),
            "measured_sci": round(hist_sci, 1) if hist_sci is not None else None,
            "fwd_score": round(hist_sci_score, 1) if hist_sci_score is not None else None,
            "fwd_health": hist_fwd_health,
            "rhi": round(hist_rhi, 1),
            "condition": hist_condition,
            "fwd_available": has_fwd and hist_sci_score is not None,
            "weights": "50% Surface + 50% Structural (Supervised AI)" if (has_fwd and hist_sci_score is not None) else "100% Surface (Fallback)",
        },
        "present_estimation": {
            "year": target_present_year,
            "estimated_iri": round(current_mri, 3),
            "iri_score": round(present_iri_score, 1),
            "estimated_sci": round(current_sci, 1) if current_sci is not None else None,
            "fwd_score": round(present_sci_score, 1) if present_sci_score is not None else None,
            "rhi": round(present_rhi, 1),
            "condition": present_condition,
            "simulated_years": max(0, target_present_year - hist_year),
            "iri_change": round(current_mri - hist_mri, 3),
            "sci_change": round(current_sci - hist_sci, 1) if (hist_sci is not None and current_sci is not None) else None,
            "policy": structural_policy,
        },
    }


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/metadata")
def metadata() -> dict[str, list[str]]:
    artifacts = load_artifacts()
    return {
        "pavement_families": artifacts["pavement_encoder"].classes_.tolist(),
        "lanes": artifacts["lane_encoder"].classes_.tolist(),
    }


@app.get("/api/sections")
def sections(search: str = Query(default="", max_length=30), limit: int = Query(default=500, ge=1, le=2000)) -> list[dict[str, str]]:
    iri_records, _ = load_network_data()
    result = iri_records[["SHRP_ID", "STATE_CODE"]].drop_duplicates()
    needle = search.strip().lower()
    if needle:
        result = result[
            result["SHRP_ID"].str.lower().str.contains(needle, na=False)
            | result["STATE_CODE"].str.lower().str.contains(needle, na=False)
        ]
    return result.sort_values(["STATE_CODE", "SHRP_ID"]).head(limit).to_dict("records")


@app.get("/api/section/{shrp_id}")
def section_detail(shrp_id: str, state_code: str = Query(...)) -> dict[str, Any]:
    iri_records, fwd_records = load_network_data()
    shrp_id = str(shrp_id).zfill(4)
    state_code = str(state_code).replace(".0", "")
    history = iri_records[(iri_records["SHRP_ID"] == shrp_id) & (iri_records["STATE_CODE"] == state_code)].copy()
    if history.empty:
        raise HTTPException(404, "No IRI history found for this SHRP_ID and STATE_CODE.")
    history = history.sort_values("YEAR")
    latest = history.iloc[-1]
    fwd_rows = fwd_records[(fwd_records["SHRP_ID"] == shrp_id) & (fwd_records["STATE_CODE"] == state_code)]
    fwd_available = not fwd_rows.empty
    defaults = {
        "mri": float(latest["MRI"]),
        "aadtt": float(latest["AADTT_ALL_TRUCKS_TREND"]),
        "annual_truck_volume": float(latest["ANNUAL_TRUCK_VOLUME_TREND"]),
        "annual_esal": float(latest["ANNUAL_ESAL_TREND"]),
        "cumulative_esal": float(latest["CUMULATIVE_ESAL"]),
        "year": int(latest["YEAR"]),
        "mean_ann_temp_avg": float(latest["MEAN_ANN_TEMP_AVG"]),
        "freeze_index_yr": float(latest["FREEZE_INDEX_YR"]),
        "freeze_thaw_yr": float(latest["FREEZE_THAW_YR"]),
        "fwd_available": fwd_available,
    }
    basin: list[float] | None = None
    if fwd_available:
        fwd_latest = fwd_rows.iloc[-1]
        basin = [float(fwd_latest[f"PEAK_DEFL_{index}"]) for index in range(1, 8)]
        defaults.update({
            "deflections": basin,
            "drop_load": float(fwd_latest["DROP_LOAD"]),
            "drop_height": int(fwd_latest["DROP_HEIGHT"]),
            "pavement_family": str(fwd_latest["PAVEMENT_FAMILY"]),
            "lane_no": str(fwd_latest["LANE_NO"]),
        })
    prediction = predict(PredictionInput(**defaults))
    return {
        "section": {"shrp_id": shrp_id, "state_code": state_code, "construction_no": str(latest["CONSTRUCTION_NO"])},
        "history": history[["YEAR", "MRI"]].to_dict("records"),
        "deflection_basin": basin,
        "deflection_confidence": {
            "lower": [round(value * 0.9, 2) for value in basin] if basin else None,
            "upper": [round(value * 1.1, 2) for value in basin] if basin else None,
        },
        "defaults": defaults,
        "prediction": prediction,
    }


@lru_cache(maxsize=1)
def compute_network_summary() -> dict[str, Any]:
    iri_records, fwd_records = load_network_data()
    artifacts = load_artifacts()
    latest = iri_records.sort_values("YEAR").groupby(["SHRP_ID", "STATE_CODE"], as_index=False).tail(1).copy()
    
    # Surface scores
    latest_iri_pred = artifacts["iri_model"].predict(latest[IRI_FEATURES])
    latest["iri_score"] = ((IRI_FAILURE_THRESHOLD - latest_iri_pred) / IRI_FAILURE_THRESHOLD * 100).clip(0, 100)

    # Merge structural if available with AASHTO BELLS temperature normalization
    fwd_records = fwd_records.copy()
    fwd_records = fwd_records.merge(
        latest[["SHRP_ID", "STATE_CODE", "MEAN_ANN_TEMP_AVG"]],
        on=["SHRP_ID", "STATE_CODE"],
        how="left",
    )
    t_pav_sum = fwd_records["MEAN_ANN_TEMP_AVG"].fillna(20.0).astype(float)
    bells_factor_sum = 10.0 ** (-0.0079 * (20.0 - t_pav_sum))
    d1_norm_sum = fwd_records["PEAK_DEFL_1"] * bells_factor_sum
    d2_norm_sum = fwd_records["PEAK_DEFL_2"] * bells_factor_sum
    fwd_records["SCI"] = d1_norm_sum - d2_norm_sum
    fwd_records["sci_score"] = ((SCI_FAILURE_THRESHOLD - fwd_records["SCI"]) / SCI_FAILURE_THRESHOLD * 100).clip(0, 100)
    fwd_summary = fwd_records.groupby(["SHRP_ID", "STATE_CODE"], as_index=False)["sci_score"].mean()

    summary = latest[["SHRP_ID", "STATE_CODE", "ANNUAL_TRUCK_VOLUME_TREND", "iri_score"]].merge(
        fwd_summary, on=["SHRP_ID", "STATE_CODE"], how="left"
    )
    summary["rhi"] = np.where(
        summary["sci_score"].isna(),
        summary["iri_score"],
        (summary["iri_score"] + summary["sci_score"]) / 2.0
    )
    summary["condition"] = summary["rhi"].map(condition)
    counts = summary["condition"].value_counts().reindex(["Good", "Fair", "Poor"], fill_value=0)
    points = summary[["SHRP_ID", "STATE_CODE", "ANNUAL_TRUCK_VOLUME_TREND", "rhi", "condition"]].rename(
        columns={"ANNUAL_TRUCK_VOLUME_TREND": "annual_truck_volume"}
    ).replace({np.nan: None}).to_dict("records")
    return {"total_sections": len(summary), "conditions": counts.to_dict(), "points": points}


@app.get("/api/network-summary")
def network_summary() -> dict[str, Any]:
    return compute_network_summary()


@app.post("/api/predict")
def live_prediction(payload: PredictionInput) -> dict[str, Any]:
    return predict(payload)


@app.post("/api/report.csv")
def download_csv(payload: PredictionInput) -> StreamingResponse:
    result = predict(payload)
    flat_data = {
        **payload.model_dump(),
        "RHI": result["rhi"],
        "Condition": result["condition"],
        "IRI_Score": result["iri_score"],
        "Predicted_Future_IRI": result["predicted_future_iri"],
        "Predicted_Future_SCI": result.get("predicted_future_sci"),
        "Structural_Score": result.get("fwd_score"),
        "Structural_Health": result.get("fwd_health"),
        "Recommendation": result["recommendation"],
    }
    content = io.StringIO()
    pd.DataFrame([flat_data]).to_csv(content, index=False)
    return StreamingResponse(
        iter([content.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=road-health-report.csv"},
    )


@app.get("/api/batch-template.csv")
def download_batch_template() -> StreamingResponse:
    template_data = [
        {
            "SHRP_ID": "0101",
            "STATE_CODE": 1,
            "YEAR": 2022,
            "MRI": 0.45,
            "AADTT_ALL_TRUCKS_TREND": 200,
            "ANNUAL_TRUCK_VOLUME_TREND": 80000,
            "ANNUAL_ESAL_TREND": 50000,
            "CUMULATIVE_ESAL": 200000,
            "MEAN_ANN_TEMP_AVG": 18.0,
            "FREEZE_INDEX_YR": 500.0,
            "FREEZE_THAW_YR": 30.0,
            "PEAK_DEFL_1": 120.0,
            "PEAK_DEFL_2": 80.0,
            "PEAK_DEFL_3": 60.0,
            "PEAK_DEFL_4": 45.0,
            "PEAK_DEFL_5": 35.0,
            "PEAK_DEFL_6": 25.0,
            "PEAK_DEFL_7": 15.0,
            "DROP_LOAD": 710.0,
            "DROP_HEIGHT": 4,
            "PAVEMENT_FAMILY": "ACUB",
            "LANE_NO": "F1",
        },
        {
            "SHRP_ID": "0102",
            "STATE_CODE": 1,
            "YEAR": 2014,
            "MRI": 0.80,
            "AADTT_ALL_TRUCKS_TREND": 700,
            "ANNUAL_TRUCK_VOLUME_TREND": 250000,
            "ANNUAL_ESAL_TREND": 200000,
            "CUMULATIVE_ESAL": 1000000,
            "MEAN_ANN_TEMP_AVG": 13.0,
            "FREEZE_INDEX_YR": 1800.0,
            "FREEZE_THAW_YR": 130.0,
            "PEAK_DEFL_1": 300.0,
            "PEAK_DEFL_2": 200.0,
            "PEAK_DEFL_3": 145.0,
            "PEAK_DEFL_4": 110.0,
            "PEAK_DEFL_5": 85.0,
            "PEAK_DEFL_6": 62.0,
            "PEAK_DEFL_7": 42.0,
            "DROP_LOAD": 710.0,
            "DROP_HEIGHT": 4,
            "PAVEMENT_FAMILY": "ACUB",
            "LANE_NO": "F3",
        },
    ]
    content = io.StringIO()
    pd.DataFrame(template_data).to_csv(content, index=False)
    return StreamingResponse(
        iter([content.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=road-batch-template.csv"},
    )


@app.post("/api/batch")
async def batch_prediction(file: UploadFile = File(...)) -> StreamingResponse:
    """Score up to 50 CSV/XLSX records with synchronized surface & structural forecasting."""
    raw = await file.read()
    try:
        frame = pd.read_csv(io.BytesIO(raw)) if file.filename.lower().endswith(".csv") else pd.read_excel(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(422, "Upload a readable CSV or Excel file.") from exc

    col_mapping = {str(col).strip().upper(): col for col in frame.columns}
    aliases = {
        "MRI": "mri",
        "AADTT_ALL_TRUCKS_TREND": "aadtt",
        "ANNUAL_TRUCK_VOLUME_TREND": "annual_truck_volume",
        "ANNUAL_ESAL_TREND": "annual_esal",
        "CUMULATIVE_ESAL": "cumulative_esal",
        "YEAR": "year",
        "MEAN_ANN_TEMP_AVG": "mean_ann_temp_avg",
        "FREEZE_INDEX_YR": "freeze_index_yr",
        "FREEZE_THAW_YR": "freeze_thaw_yr",
    }
    alt_aliases = {
        "AADTT": "aadtt",
        "ANNUAL_TRUCK_VOLUME": "annual_truck_volume",
        "ANNUAL_ESAL": "annual_esal",
        "MEAN_TEMP": "mean_ann_temp_avg",
        "FREEZE_INDEX": "freeze_index_yr",
        "FREEZE_THAW": "freeze_thaw_yr",
    }

    rename_dict = {}
    for standard_name, target in aliases.items():
        if standard_name in col_mapping:
            rename_dict[col_mapping[standard_name]] = target
        else:
            for alt_name, alt_target in alt_aliases.items():
                if alt_target == target and alt_name in col_mapping:
                    rename_dict[col_mapping[alt_name]] = target
                    break

    frame = frame.rename(columns=rename_dict)
    required = list(aliases.values())
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise HTTPException(422, f"Missing required columns: {', '.join(missing)}")

    results = []
    for index, row in frame.head(50).iterrows():
        try:
            fwd_cols_upper = {str(col).strip().upper(): col for col in row.index}
            defl_keys = [f"PEAK_DEFL_{i}" for i in range(1, 8)]
            has_all_defls = all(key in fwd_cols_upper and pd.notna(row[fwd_cols_upper[key]]) and str(row[fwd_cols_upper[key]]).strip() != "" for key in defl_keys)
            has_drop_load = "DROP_LOAD" in fwd_cols_upper and pd.notna(row[fwd_cols_upper["DROP_LOAD"]]) and str(row[fwd_cols_upper["DROP_LOAD"]]).strip() != ""
            has_drop_height = "DROP_HEIGHT" in fwd_cols_upper and pd.notna(row[fwd_cols_upper["DROP_HEIGHT"]]) and str(row[fwd_cols_upper["DROP_HEIGHT"]]).strip() != ""
            has_pav = "PAVEMENT_FAMILY" in fwd_cols_upper and pd.notna(row[fwd_cols_upper["PAVEMENT_FAMILY"]]) and str(row[fwd_cols_upper["PAVEMENT_FAMILY"]]).strip() != ""
            has_lane = "LANE_NO" in fwd_cols_upper and pd.notna(row[fwd_cols_upper["LANE_NO"]]) and str(row[fwd_cols_upper["LANE_NO"]]).strip() != ""

            fwd_params = {}
            if has_all_defls and has_drop_load and has_drop_height and has_pav and has_lane:
                deflections = [float(row[fwd_cols_upper[f"PEAK_DEFL_{i}"]]) for i in range(1, 8)]
                fwd_params = {
                    "fwd_available": True,
                    "deflections": deflections,
                    "drop_load": float(row[fwd_cols_upper["DROP_LOAD"]]),
                    "drop_height": int(row[fwd_cols_upper["DROP_HEIGHT"]]),
                    "pavement_family": str(row[fwd_cols_upper["PAVEMENT_FAMILY"]]),
                    "lane_no": str(row[fwd_cols_upper["LANE_NO"]]),
                }
            else:
                fwd_params = {"fwd_available": False}

            pred_input = PredictionInput(
                mri=float(row["mri"]),
                aadtt=float(row["aadtt"]),
                annual_truck_volume=float(row["annual_truck_volume"]),
                annual_esal=float(row["annual_esal"]),
                cumulative_esal=float(row["cumulative_esal"]),
                year=int(row["year"]),
                mean_ann_temp_avg=float(row["mean_ann_temp_avg"]),
                freeze_index_yr=float(row["freeze_index_yr"]),
                freeze_thaw_yr=float(row["freeze_thaw_yr"]),
                **fwd_params,
            )
            result = predict(pred_input)

            shrp_val = row.get("SHRP_ID", row.get("shrp_id", f"Row-{index + 1}"))
            state_val = row.get("STATE_CODE", row.get("state_code", ""))

            hist = result["historical_snapshot"]
            pres = result["present_estimation"]

            results.append({
                "Row": index + 1,
                "SHRP_ID": shrp_val,
                "STATE_CODE": state_val,
                "Survey_Year": hist["year"],
                "Historical_IRI": hist["measured_iri"],
                "Historical_IRI_Score": hist["iri_score"],
                "Historical_SCI": hist["measured_sci"] if hist["measured_sci"] is not None else "N/A",
                "Historical_Structural_Score": hist["fwd_score"] if hist["fwd_score"] is not None else "N/A",
                "Historical_RHI": hist["rhi"],
                "Historical_Condition": hist["condition"],
                "Present_Year": pres["year"],
                "Estimated_2026_IRI": pres["estimated_iri"],
                "Present_2026_IRI_Score": pres["iri_score"],
                "Estimated_2026_SCI": pres["estimated_sci"] if pres["estimated_sci"] is not None else "N/A",
                "Present_2026_Structural_Score": pres["fwd_score"] if pres["fwd_score"] is not None else "N/A",
                "Present_2026_RHI": pres["rhi"],
                "Present_2026_Condition": pres["condition"],
                "Fast_Forward_Years": pres["simulated_years"],
                "Structural_Policy": pres["policy"],
                "Recommendation": result["recommendation"],
            })
        except Exception as exc:
            results.append({
                "Row": index + 1,
                "SHRP_ID": row.get("SHRP_ID", row.get("shrp_id", f"Row-{index + 1}")),
                "STATE_CODE": row.get("STATE_CODE", row.get("state_code", "")),
                "Error": str(exc),
            })

    content = io.StringIO()
    pd.DataFrame(results).to_csv(content, index=False)
    return StreamingResponse(
        iter([content.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch-rhi-results.csv"},
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
