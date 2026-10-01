"""Comprehensive Automated Test Suite for Road RSL Prediction Architecture.

Tests cover:
1. Artifacts & Models Integrity
2. API Endpoint Functionality & Contracts
3. Data Validation & Boundary Conditions
4. Dual-Track Mechanics (Model 1 IRI, Model 2 SCI with BELLS correction)
5. Continuous Lifecycle & Virtual Maintenance Trigger
6. Batch Processing (CSV & Excel)
7. Static File & Frontend Integration
"""

import io
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from Dashboard.main import app, MODEL_DIR, PredictionInput, predict, score_iri, score_sci, condition

client = TestClient(app)


# ==============================================================================
# 1. ARTIFACTS & MODEL INTEGRITY TESTS
# ==============================================================================
class TestArtifacts:
    def test_required_artifacts_exist(self):
        """Verify all model binaries and calibration rates are present."""
        expected_files = [
            "iri_prediction_model.pkl",
            "sci_prediction_model.pkl",
            "sci_le_pav.pkl",
            "sci_le_lane.pkl",
            "deterioration_rate.txt",
            "sci_deterioration_rate.txt",
        ]
        for filename in expected_files:
            file_path = MODEL_DIR / filename
            assert file_path.exists(), f"Missing required artifact: {filename}"

    def test_fallback_rates_calibration(self):
        """Verify fallback rates are calibrated within AASHTO physics bounds."""
        sci_rate_file = MODEL_DIR / "sci_deterioration_rate.txt"
        with open(sci_rate_file, "r") as f:
            sci_rate = float(f.read().strip())
        # Calibrated default is 4.2 um/yr, strictly not legacy 19.0
        assert sci_rate == 4.2, f"Expected calibrated rate 4.2 um/yr, found {sci_rate}"

        iri_rate_file = MODEL_DIR / "deterioration_rate.txt"
        with open(iri_rate_file, "r") as f:
            iri_rate = float(f.read().strip())
        assert 0.01 <= iri_rate <= 0.10, f"IRI deterioration rate {iri_rate} out of bounds"


# ==============================================================================
# 2. REST API CONTRACT TESTS
# ==============================================================================
class TestApiEndpoints:
    def test_health_check(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"
        assert "version" in data

    def test_metadata(self):
        resp = client.get("/api/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert "pavement_families" in data
        assert "lanes" in data
        assert len(data["pavement_families"]) > 0
        assert len(data["lanes"]) > 0

    def test_sections_search(self):
        # Unfiltered
        resp = client.get("/api/sections?limit=10")
        assert resp.status_code == 200
        sections = resp.json()
        assert isinstance(sections, list)
        assert len(sections) <= 10
        assert len(sections) > 0
        assert "SHRP_ID" in sections[0]
        assert "STATE_CODE" in sections[0]

        # Search filter
        sample_id = sections[0]["SHRP_ID"]
        resp_filtered = client.get(f"/api/sections?search={sample_id}")
        assert resp_filtered.status_code == 200
        filtered = resp_filtered.json()
        assert any(s["SHRP_ID"] == sample_id for s in filtered)

    def test_section_detail_valid(self):
        # Fetch first valid section
        sections_resp = client.get("/api/sections?limit=1")
        assert sections_resp.status_code == 200
        sec = sections_resp.json()[0]
        shrp_id = sec["SHRP_ID"]
        state_code = sec["STATE_CODE"]

        resp = client.get(f"/api/section/{shrp_id}?state_code={state_code}")
        assert resp.status_code == 200
        detail = resp.json()
        assert "section" in detail
        assert "history" in detail
        assert "defaults" in detail
        assert "prediction" in detail

    def test_section_detail_invalid(self):
        resp = client.get("/api/section/999999?state_code=99")
        assert resp.status_code == 404

    def test_network_summary(self):
        resp = client.get("/api/network-summary")
        assert resp.status_code == 200
        summary = resp.json()
        assert "total_sections" in summary
        assert "conditions" in summary
        assert "points" in summary
        assert summary["total_sections"] > 0
        assert set(summary["conditions"].keys()) == {"Good", "Fair", "Poor"}


# ==============================================================================
# 3. CORE PREDICTION & MATHEMATICAL MECHANICS TESTS
# ==============================================================================
class TestPredictionMechanics:
    @pytest.fixture
    def base_fwd_payload(self):
        return {
            "mri": 0.85,
            "aadtt": 950,
            "annual_truck_volume": 346750,
            "annual_esal": 310000,
            "cumulative_esal": 1500000,
            "year": 2015,
            "mean_ann_temp_avg": 15.5,
            "freeze_index_yr": 10,
            "freeze_thaw_yr": 45,
            "fwd_available": True,
            "deflections": [450, 280, 210, 180, 140, 110, 70],
            "drop_load": 710.0,
            "drop_height": 4,
            "pavement_family": "ACUB",
            "lane_no": "F1",
        }

    def test_full_dual_prediction(self, base_fwd_payload):
        resp = client.post("/api/predict", json=base_fwd_payload)
        assert resp.status_code == 200
        res = resp.json()
        assert 0 <= res["rhi"] <= 100
        assert res["condition"] in ["Good", "Fair", "Poor"]
        assert res["fwd_score"] is not None
        assert res["predicted_future_sci"] is not None
        assert len(res["simulation_path"]) == (2026 - 2015) + 1
        assert len(res["projection"]) == 10
        assert len(res["explanation"]) > 0

    def test_surface_only_prediction(self, base_fwd_payload):
        payload = base_fwd_payload.copy()
        payload["fwd_available"] = False
        payload["deflections"] = None

        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 200
        res = resp.json()
        assert res["fwd_score"] is None
        assert res["predicted_future_sci"] is None
        # RHI should exactly equal IRI score under 100% surface fallback
        assert abs(res["rhi"] - res["iri_score"]) < 1e-1
        assert "100% Surface" in res["historical_snapshot"]["weights"]

    def test_bells_temperature_normalization(self, base_fwd_payload):
        """Verify BELLS exponential correction properly adjusts deflections."""
        t_pav = 15.5
        expected_factor = 10.0 ** (-0.0079 * (20.0 - t_pav))
        d1 = base_fwd_payload["deflections"][0]
        d2 = base_fwd_payload["deflections"][1]
        expected_sci = (d1 * expected_factor) - (d2 * expected_factor)

        resp = client.post("/api/predict", json=base_fwd_payload)
        res = resp.json()
        measured_sci = res["historical_snapshot"]["measured_sci"]
        assert round(measured_sci, 1) == round(expected_sci, 1)

    def test_virtual_maintenance_trigger(self):
        """Verify virtual maintenance overlays activate when SCI > 150 um."""
        severe_fwd_payload = {
            "mri": 1.2,
            "aadtt": 1500,
            "annual_truck_volume": 500000,
            "annual_esal": 400000,
            "cumulative_esal": 3000000,
            "year": 2010,
            "mean_ann_temp_avg": 22.0,
            "freeze_index_yr": 5,
            "freeze_thaw_yr": 20,
            "fwd_available": True,
            "deflections": [600, 460, 300, 200, 150, 100, 60],
            "drop_load": 710.0,
            "drop_height": 4,
            "pavement_family": "ACUB",
            "lane_no": "F1",
        }
        resp = client.post("/api/predict", json=severe_fwd_payload)
        assert resp.status_code == 200
        res = resp.json()
        path = res["simulation_path"]

        # Check if overlay reset occurs at 40.0 um
        sci_values = [p["sci"] for p in path if p["sci"] is not None]
        assert any(v == 40.0 for v in sci_values), "Virtual maintenance trigger did not reset SCI to 40.0 um"

        # Check no simulated SCI in the path ever exceeds 150 um
        assert all(v <= 150.0 for v in sci_values), "Simulated SCI exceeded critical 150 um threshold without overlay"

    def test_bounded_degradation_envelope(self, base_fwd_payload):
        """Verify annual deterioration remains inside the AASHTO calibrated bounds (1.5 to 8.0 um/yr)."""
        resp = client.post("/api/predict", json=base_fwd_payload)
        res = resp.json()
        path = res["simulation_path"]

        for i in range(1, len(path)):
            prev_sci = path[i - 1]["sci"]
            curr_sci = path[i]["sci"]
            if prev_sci is not None and curr_sci is not None and curr_sci != 40.0:
                delta = curr_sci - prev_sci
                # Delta must be within physical bounds
                assert 1.4 <= delta <= 8.1, f"Annual delta {delta} outside calibrated envelope [1.5, 8.0]"


# ==============================================================================
# 4. INPUT VALIDATION & BOUNDARY TESTS
# ==============================================================================
class TestInputValidation:
    def test_year_boundary(self):
        # Under minimum 1980
        resp_low = client.post("/api/predict", json={"year": 1970, "mri": 1.0, "aadtt": 100, "annual_truck_volume": 1000, "annual_esal": 1000, "cumulative_esal": 1000, "mean_ann_temp_avg": 15, "freeze_index_yr": 10, "freeze_thaw_yr": 10})
        assert resp_low.status_code == 422

        # Above maximum 2030
        resp_high = client.post("/api/predict", json={"year": 2040, "mri": 1.0, "aadtt": 100, "annual_truck_volume": 1000, "annual_esal": 1000, "cumulative_esal": 1000, "mean_ann_temp_avg": 15, "freeze_index_yr": 10, "freeze_thaw_yr": 10})
        assert resp_high.status_code == 422

    def test_deflections_length_validation(self):
        # 5 deflections instead of 7
        resp = client.post("/api/predict", json={
            "mri": 0.8, "aadtt": 100, "annual_truck_volume": 1000, "annual_esal": 1000,
            "cumulative_esal": 1000, "year": 2020, "mean_ann_temp_avg": 15,
            "freeze_index_yr": 10, "freeze_thaw_yr": 10, "fwd_available": True,
            "deflections": [100, 80, 60, 40, 20]
        })
        assert resp.status_code == 422

    def test_deflection_range_validation(self):
        # Negative deflection
        resp = client.post("/api/predict", json={
            "mri": 0.8, "aadtt": 100, "annual_truck_volume": 1000, "annual_esal": 1000,
            "cumulative_esal": 1000, "year": 2020, "mean_ann_temp_avg": 15,
            "freeze_index_yr": 10, "freeze_thaw_yr": 10, "fwd_available": True,
            "deflections": [-10, 80, 60, 40, 20, 10, 5]
        })
        assert resp.status_code == 422


# ==============================================================================
# 5. REPORTS & BATCH PROCESSING TESTS
# ==============================================================================
class TestReportingAndBatch:
    def test_csv_report_generation(self):
        payload = {
            "mri": 0.9, "aadtt": 800, "annual_truck_volume": 300000, "annual_esal": 250000,
            "cumulative_esal": 1200000, "year": 2018, "mean_ann_temp_avg": 16.0,
            "freeze_index_yr": 15, "freeze_thaw_yr": 35, "fwd_available": False,
        }
        resp = client.post("/api/report.csv", json=payload)
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        content = resp.text
        assert "RHI" in content
        assert "Condition" in content

    def test_batch_template_download(self):
        resp = client.get("/api/batch-template.csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        df = pd.read_csv(io.StringIO(resp.text))
        required_cols = ["SHRP_ID", "STATE_CODE", "YEAR", "MRI", "AADTT_ALL_TRUCKS_TREND", "PEAK_DEFL_1"]
        for col in required_cols:
            assert col in df.columns

    def test_batch_csv_upload_success(self):
        template_resp = client.get("/api/batch-template.csv")
        csv_bytes = template_resp.content

        files = {"file": ("test_batch.csv", csv_bytes, "text/csv")}
        resp = client.post("/api/batch", files=files)
        assert resp.status_code == 200
        df_result = pd.read_csv(io.StringIO(resp.text))
        assert "Present_2026_RHI" in df_result.columns
        assert len(df_result) == 2


# ==============================================================================
# 6. STATIC FRONTEND SERVING TEST
# ==============================================================================
class TestFrontendIntegration:
    def test_dashboard_index_serves_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        html = resp.text
        assert "Road Health Index" in html
        assert "app.js" in html
