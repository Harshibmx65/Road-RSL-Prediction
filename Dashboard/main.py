"""FastAPI dashboard for the Road RSL Prediction project.

This app is deliberately self-contained: it reads the existing project data and
model artifacts but never writes to them. Run it from the Dashboard directory.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
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
FWD_FEATURES = [
    "PEAK_DEFL_1", "PEAK_DEFL_2", "PEAK_DEFL_3", "PEAK_DEFL_4", "PEAK_DEFL_5",
    "PEAK_DEFL_6", "PEAK_DEFL_7", "DROP_LOAD", "DROP_HEIGHT",
    "PAVEMENT_FAMILY_ENC", "LANE_NO_ENC",
]
FWD_SCORE_MAP = {"Good": 100, "Fair": 60, "Poor": 20}
FAILURE_THRESHOLD = 2.5


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
    # The model was trained with the source workbook's native scale (e.g., 710).
    drop_load: float | None = Field(default=None, ge=0, le=2_000)
    drop_height: int | None = Field(default=None, ge=1, le=4)
    pavement_family: str | None = None
    lane_no: str | None = None

    @field_validator("year")
    @classmethod
    def validate_year(cls, value: int) -> int:
        if value < 1980 or value > 2030:
            raise ValueError("Year must be between 1980 and 2030.")
        if value > datetime.now().year:
            raise ValueError("Year cannot be greater than the current year.")
        return value

    @field_validator("deflections")
    @classmethod
    def validate_deflections(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        if len(value) != 7:
            raise ValueError("Provide exactly seven FWD deflection values.")
        if any(item < 0 or item > 2_000 for item in value):
            raise ValueError("Each FWD deflection must be between 0 and 2,000 microns.")
        return value


app = FastAPI(title="Road Health Index Dashboard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("DASHBOARD_ALLOWED_ORIGIN", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.zfill(4)


def condition(score: float) -> str:
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"


def recommendation(score: float) -> str:
    if score >= 75:
        return "Routine inspection and preventive maintenance."
    if score >= 50:
        return "Plan maintenance and investigate the contributing component."
    return "Prioritize detailed inspection and corrective maintenance."


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    required = {
        "iri_model": "iri_prediction_model.pkl",
        "kmeans": "fwd_kmeans_model.pkl",
        "scaler": "fwd_scaler.pkl",
        "pavement_encoder": "fwd_le_pav.pkl",
        "lane_encoder": "fwd_le_lane.pkl",
        "health_mapping": "fwd_health_mapping.pkl",
    }
    missing = [filename for filename in required.values() if not (MODEL_DIR / filename).exists()]
    if missing:
        raise HTTPException(503, f"Missing model artifacts: {', '.join(missing)}. Train the models first.")
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in required.items()}


@lru_cache(maxsize=1)
def load_network_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return prepared IRI records and FWD records from the source workbooks."""
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
    required_fwd = [*FWD_FEATURES[:9], "PAVEMENT_FAMILY", "LANE_NO"]
    fwd_records = fwd_records.dropna(subset=required_fwd).copy()
    return iri_records, fwd_records


def predict(payload: PredictionInput) -> dict[str, Any]:
    artifacts = load_artifacts()
    iri_input = pd.DataFrame([[
        payload.mri, payload.aadtt, payload.annual_truck_volume, payload.annual_esal,
        payload.cumulative_esal, payload.year, payload.mean_ann_temp_avg,
        payload.freeze_index_yr, payload.freeze_thaw_yr,
    ]], columns=IRI_FEATURES)
    future_iri = float(artifacts["iri_model"].predict(iri_input)[0])
    iri_score = float(np.clip(((FAILURE_THRESHOLD - future_iri) / FAILURE_THRESHOLD) * 100, 0, 100))

    fwd_score: float | None = None
    health: str | None = None
    fallback = not payload.fwd_available
    if payload.fwd_available:
        if not payload.deflections or len(payload.deflections) != 7:
            raise HTTPException(422, "Provide exactly seven FWD deflection values when FWD data is available.")
        if any(value < 0 or value > 2_000 for value in payload.deflections):
            raise HTTPException(422, "Each FWD deflection must be between 0 and 2,000 microns.")
        if None in (payload.drop_load, payload.drop_height, payload.pavement_family, payload.lane_no):
            raise HTTPException(422, "Complete all FWD fields or turn off FWD availability.")
        try:
            pavement = artifacts["pavement_encoder"].transform([payload.pavement_family])[0]
            lane = artifacts["lane_encoder"].transform([payload.lane_no])[0]
        except ValueError as exc:
            raise HTTPException(422, f"Unsupported FWD category: {exc}") from exc
        fwd_input = pd.DataFrame([[
            *payload.deflections, payload.drop_load, payload.drop_height, pavement, lane,
        ]], columns=FWD_FEATURES)
        cluster = int(artifacts["kmeans"].predict(artifacts["scaler"].transform(fwd_input))[0])
        health = artifacts["health_mapping"][cluster]
        fwd_score = float(FWD_SCORE_MAP[health])

    rhi = iri_score if fallback else (iri_score + fwd_score) / 2
    contributions = artifacts["iri_model"].get_booster().predict(
        DMatrix(iri_input, feature_names=IRI_FEATURES), pred_contribs=True
    )[0][:-1]
    total_impact = sum(abs(float(value)) for value in contributions) or 1
    explanation = sorted([
        {"feature": feature.replace("_", " ").title(), "impact_percent": round(abs(float(value)) / total_impact * 100, 1),
         "direction": "increases roughness risk" if value > 0 else "reduces roughness risk"}
        for feature, value in zip(IRI_FEATURES, contributions)
    ], key=lambda item: item["impact_percent"], reverse=True)[:4]
    projected_iri = payload.mri
    projection = []
    for offset in range(1, 11):
        future_year = payload.year + offset
        future_input = pd.DataFrame([[projected_iri, payload.aadtt, payload.annual_truck_volume,
            payload.annual_esal, payload.cumulative_esal + payload.annual_esal * offset, future_year,
            payload.mean_ann_temp_avg, payload.freeze_index_yr, payload.freeze_thaw_yr]], columns=IRI_FEATURES)
        projected_iri = float(artifacts["iri_model"].predict(future_input)[0])
        projection.append({"year": future_year, "iri": round(projected_iri, 3), "iri_score": round(float(np.clip(((FAILURE_THRESHOLD - projected_iri) / FAILURE_THRESHOLD) * 100, 0, 100)), 2)})
    return {
        "predicted_future_iri": round(future_iri, 3),
        "iri_score": round(iri_score, 2),
        "fwd_score": fwd_score,
        "fwd_health": health,
        "rhi": round(rhi, 2),
        "condition": condition(rhi),
        "recommendation": recommendation(rhi),
        "fallback_engaged": fallback,
        "explanation": explanation,
        "projection": projection,
    }


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metadata")
def metadata() -> dict[str, list[str]]:
    artifacts = load_artifacts()
    return {
        "pavement_families": artifacts["pavement_encoder"].classes_.tolist(),
        "lanes": artifacts["lane_encoder"].classes_.tolist(),
    }


@app.get("/api/sections")
def sections(search: str = Query(default="", max_length=30), limit: int = Query(default=25, ge=1, le=100)) -> list[dict[str, str]]:
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
        "mri": float(latest["MRI"]), "aadtt": float(latest["AADTT_ALL_TRUCKS_TREND"]),
        "annual_truck_volume": float(latest["ANNUAL_TRUCK_VOLUME_TREND"]),
        "annual_esal": float(latest["ANNUAL_ESAL_TREND"]), "cumulative_esal": float(latest["CUMULATIVE_ESAL"]),
        "year": int(latest["YEAR"]), "mean_ann_temp_avg": float(latest["MEAN_ANN_TEMP_AVG"]),
        "freeze_index_yr": float(latest["FREEZE_INDEX_YR"]), "freeze_thaw_yr": float(latest["FREEZE_THAW_YR"]),
        "fwd_available": fwd_available,
    }
    basin: list[float] | None = None
    if fwd_available:
        fwd_latest = fwd_rows.iloc[-1]
        basin = [float(fwd_latest[f"PEAK_DEFL_{index}"]) for index in range(1, 8)]
        defaults.update({
            "deflections": basin, "drop_load": float(fwd_latest["DROP_LOAD"]),
            "drop_height": int(fwd_latest["DROP_HEIGHT"]),
            "pavement_family": str(fwd_latest["PAVEMENT_FAMILY"]), "lane_no": str(fwd_latest["LANE_NO"]),
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


@app.get("/api/network-summary")
def network_summary() -> dict[str, Any]:
    iri_records, fwd_records = load_network_data()
    latest = iri_records.sort_values("YEAR").groupby(["SHRP_ID", "STATE_CODE"], as_index=False).tail(1)
    score = ((FAILURE_THRESHOLD - load_artifacts()["iri_model"].predict(latest[IRI_FEATURES])) / FAILURE_THRESHOLD * 100).clip(0, 100)
    summary = latest[["SHRP_ID", "STATE_CODE"]].copy()
    summary["iri_score"] = score
    artifacts = load_artifacts()
    fwd_records = fwd_records.copy()
    fwd_records["PAVEMENT_FAMILY_ENC"] = artifacts["pavement_encoder"].transform(fwd_records["PAVEMENT_FAMILY"])
    fwd_records["LANE_NO_ENC"] = artifacts["lane_encoder"].transform(fwd_records["LANE_NO"])
    clusters = artifacts["kmeans"].predict(artifacts["scaler"].transform(fwd_records[FWD_FEATURES]))
    fwd_records["fwd_score"] = pd.Series(clusters, index=fwd_records.index).map(artifacts["health_mapping"]).map(FWD_SCORE_MAP)
    fwd_scores = fwd_records.groupby(["SHRP_ID", "STATE_CODE"], as_index=False)["fwd_score"].mean()
    summary["annual_truck_volume"] = latest["ANNUAL_TRUCK_VOLUME_TREND"].values
    summary = summary.merge(fwd_scores, on=["SHRP_ID", "STATE_CODE"], how="left")
    summary["rhi"] = np.where(summary["fwd_score"].isna(), summary["iri_score"], (summary["iri_score"] + summary["fwd_score"]) / 2)
    summary["condition"] = summary["rhi"].map(condition)
    counts = summary["condition"].value_counts().reindex(["Good", "Fair", "Poor"], fill_value=0)
    points = summary[["SHRP_ID", "STATE_CODE", "annual_truck_volume", "rhi", "condition"]].replace({np.nan: None}).to_dict("records")
    return {"total_sections": len(summary), "conditions": counts.to_dict(), "points": points}


@app.post("/api/predict")
def live_prediction(payload: PredictionInput) -> dict[str, Any]:
    return predict(payload)


@app.post("/api/report.csv")
def download_csv(payload: PredictionInput) -> StreamingResponse:
    result = predict(payload)
    report = pd.DataFrame([{**payload.model_dump(), **result}])
    content = io.StringIO()
    report.to_csv(content, index=False)
    return StreamingResponse(
        iter([content.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=road-health-report.csv"},
    )


@app.post("/api/batch")
async def batch_prediction(file: UploadFile = File(...)) -> StreamingResponse:
    """Score up to 50 CSV/XLSX records. Required columns are documented in the UI."""
    raw = await file.read()
    try:
        frame = pd.read_csv(io.BytesIO(raw)) if file.filename.lower().endswith(".csv") else pd.read_excel(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(422, "Upload a readable CSV or Excel file.") from exc
    aliases = {"MRI": "mri", "AADTT_ALL_TRUCKS_TREND": "aadtt", "ANNUAL_TRUCK_VOLUME_TREND": "annual_truck_volume", "ANNUAL_ESAL_TREND": "annual_esal", "CUMULATIVE_ESAL": "cumulative_esal", "YEAR": "year"}
    frame = frame.rename(columns=aliases)
    required = list(aliases.values())
    missing = [column for column in required if column not in frame]
    if missing:
        raise HTTPException(422, f"Missing required columns: {', '.join(missing)}")
    results = []
    for index, row in frame.head(50).iterrows():
        try:
            result = predict(PredictionInput(**{column: row[column] for column in required}, fwd_available=False))
            results.append({"row": index + 1, **result})
        except Exception as exc:
            results.append({"row": index + 1, "error": str(exc)})
    content = io.StringIO()
    pd.DataFrame(results).to_csv(content, index=False)
    return StreamingResponse(iter([content.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=batch-rhi-results.csv"})


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
