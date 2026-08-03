# Predicting Pavement Condition and Road Health Index 🛣️

**Group 18 | Department of AI & ML | Acharya Institute of Technology**

**Guide**: Mr. Mohammed Tahir Mirji | Assistant Professor

---

## 1. Project Overview

### What This Project Does

This project builds a machine learning system that predicts the future surface condition and structural health of road pavements using Non-Destructive Testing (NDT) data. NDT means we measure the road without damaging it, utilizing sensors, falling weight deflectometers, and profiling instruments.

### Why It Matters

* Roads deteriorate over time due to traffic load and environmental conditions.
* Manual inspection is time-consuming and costly.
* This system allows road authorities to evaluate road health algorithmically at scale.
* Enables proactive maintenance instead of reactive repair.

### Two-Model Architecture

The system employs a dual-track machine learning architecture (supervised and unsupervised) that fuses surface roughness and structural integrity into a single, comprehensive Road Health Index (RHI).

---

## 2. Datasets

All data comes from the LTPP (Long-Term Pavement Performance) database maintained by the US Federal Highway Administration (FHWA).

**Common Key Columns:**

* `SHRP_ID`: Unique road section identifier (e.g., 0101, 0102)
* `STATE_CODE`: US state number (1 = Alabama)
* `CONSTRUCTION_NO`: Construction version of that section (1 = original, 2 = after repair)

---

## 3. Model 1 - IRI + Traffic (Surface Deterioration)

**Purpose**: Predict the future International Roughness Index (IRI) of a road section and convert that prediction into a normalized 0-100 IRI Score.

**Algorithm**: XGBoost Regressor (Supervised Learning)

**Data Engineering**: Implements forward-fill imputation for temporal traffic trends to prevent data loss and retain maximum training records.

**Features Used**:

* `MRI` (Current road roughness)
* `AADTT_ALL_TRUCKS_TREND` (Daily truck count)
* `ANNUAL_TRUCK_VOLUME_TREND` (Yearly total truck volume)
* `ANNUAL_ESAL_TREND` (Yearly damage load)
* `CUMULATIVE_ESAL` (Engineered: Total damage load since the road was built)
* `YEAR` (Year of measurement)

---

## 4. Model 2 - FWD (Structural Health)

**Purpose**: Discover structural health patterns and assign Good, Fair, or Poor classifications dynamically based on Falling Weight Deflectometer (FWD) measurements.

**Algorithm**: K-Means Clustering (Unsupervised Learning)

**Data Engineering**: Replaced the original supervised classifier with an unsupervised clustering model to completely eliminate data leakage. Measurements are standardized using `StandardScaler` to naturally group roads by structural similarity via centroid analysis.

**Features Used**:

* `PEAK_DEFL_1` to `PEAK_DEFL_7` (Deflections at varying sensor distances)
* `DROP_LOAD` (Load applied during test)
* `DROP_HEIGHT` (Height of drop 1-4)
* `PAVEMENT_FAMILY_ENC` (Encoded type of pavement construction)
* `LANE_NO_ENC` (Encoded lane position tested)

---

## 5. RHI - Road Health Index

**Purpose**: Combine Model 1 (surface condition) and Model 2 (structural health) into a single score (0-100) that represents overall road health.

**Dynamic Fallback Architecture**:
The system is built for real-world scalability. It utilizes a Left Join to evaluate all road sections.

* If both IRI and FWD data are present: **RHI = 50% IRI Score + 50% FWD Score**.
* If FWD data is missing (due to incomplete sensor records): **RHI = 100% IRI Score** (Dynamic Fallback Engaged).

| RHI Score | Condition | Recommended Action |
| --- | --- | --- |
| 75-100 | Good | Routine maintenance only |
| 50-74 | Fair | Schedule repairs in near future |
| 0-49 | Poor | Immediate attention required |

---

## 6. How to Use for a New Road

To predict the RHI for any new road, execute the interactive RHI predictor script. You will be prompted to provide the required IRI and traffic data. The script will then ask if FWD structural data is available; if you select "no", the system will automatically engage the dynamic fallback logic and output the final index.

---

## 7. Road Health Index Dashboard

The project includes a FastAPI dashboard that reads the existing `data` and `models` artifacts without modifying them.

### Run the dashboard

From the project root, create and activate a virtual environment, install the unified dependencies, then start the dashboard:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn Dashboard.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

### REST endpoints

* `GET /api/sections?search=0101` — section search
* `GET /api/section/{shrp_id}?state_code={state}` — historical records, defaults, and prediction
* `GET /api/network-summary` — condition distribution
* `POST /api/predict` — live what-if prediction
* `POST /api/report.csv` — download a prediction report

The dashboard generates formatted PDFs client-side. MySQL is not needed for local use because the dashboard reads the existing Excel source data. A database adapter is only needed for persistent user accounts, saved scenarios, or multi-user deployment.

---

## 8. Glossary

* **NDT**: Non-Destructive Testing
* **IRI**: International Roughness Index (m/km)
* **FWD**: Falling Weight Deflectometer
* **RHI**: Road Health Index
* **ESAL**: Equivalent Single Axle Load
* **LTPP**: Long-Term Pavement Performance
* **K-Means**: An unsupervised machine learning algorithm used to cluster unlabeled data.
