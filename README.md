# Predicting Pavement Condition and Remaining Service Life 🛣️
**Group 18 | Department of AI & ML | Acharya Institute of Technology**  
**Guide**: Mr. Mohammed Tahir Mirji | Assistant Professor

---

## 1. Project Overview
### What This Project Does
This project builds a machine learning system that predicts the health condition and remaining useful life (RUL) of road pavements using Non-Destructive Testing (NDT) data. NDT means we measure the road without damaging it - using sensors, radar, and instruments.

### Why It Matters
* Roads deteriorate over time due to traffic load and environmental conditions.
* Manual inspection is time-consuming and costly.
* This system allows road authorities to predict when a road will fail - before it actually fails.
* Enables proactive maintenance instead of reactive repair.

### Two-Model Architecture
The system uses two separate ML models that are combined into one final score:

| Model | Input | Output |
|-------|-------|--------|
| **Model 1** | Road roughness + traffic load over time | Predicted future IRI + RUL (years) |
| **Model 2** | FWD deflection measurements | Structural health (Good/Fair/Poor) |
| **RHI** | IRI Score + FWD Score | Single health score (0-100) |

---

## 2. Datasets
All data comes from the LTPP (Long-Term Pavement Performance) database maintained by the US Federal Highway Administration (FHWA).

**Common Key Columns:**
* `SHRP_ID`: Unique road section identifier (e.g., 0101, 0102)
* `STATE_CODE`: US state number (1 = Alabama)
* `CONSTRUCTION_NO`: Construction version of that section (1 = original, 2 = after repair)

---

## 3. Model 1 - IRI + Traffic (Surface Deterioration)
**Purpose**: Predict the future IRI (road roughness at next measurement) and derive RUL (Remaining Useful Life in years) from it.

**Algorithm**: XGBoost Regressor ($R^2$ Score: 0.9411)

**Features Used**:
* `MRI` (Current road roughness)
* `AADTT_ALL_TRUCKS_TREND` (Daily truck count)
* `ANNUAL_TRUCK_VOLUME_TREND` (Yearly total truck volume)
* `ANNUAL_ESAL_TREND` (Yearly damage load)
* `CUMULATIVE_ESAL` (Engineered: Total damage since road built)
* `IRI_GROWTH_RATE` (Engineered: Rate of IRI increase per year)
* `YEAR` (Year of measurement)

---

## 4. Model 2 - FWD (Structural Health)
**Purpose**: Classify the structural health of a road section as Good, Fair, or Poor based on FWD deflection measurements and pavement type.

**Algorithm**: XGBoost Classifier (Accuracy: 0.9982)

**Features Used**:
* `PEAK_DEFL_1` to `PEAK_DEFL_7` (Deflections at varying sensor distances)
* `DROP_LOAD` (Load applied during test)
* `DROP_HEIGHT` (Height of drop 1-4)
* `PAVEMENT_FAMILY_ENC` (Encoded type of pavement construction)
* `LANE_NO_ENC` (Encoded lane position tested)

---

## 5. RHI - Road Health Index
**Purpose**: Combine Model 1 (surface condition) and Model 2 (structural health) into a single score (0-100) that represents overall road health.

| RHI Score | Condition | Recommended Action |
|-----------|-----------|--------------------|
| 75-100 | Good | Routine maintenance only |
| 50-74 | Fair | Schedule repairs in near future |
| 0-49 | Poor | Immediate attention required |

---

## 6. How to Use for a New Road
To predict RHI for any new road, run the RHI predictor and provide the required IRI, traffic, and FWD inputs when prompted. 

---

## 7. Glossary
* **NDT**: Non-Destructive Testing
* **IRI**: International Roughness Index ($m/km$)
* **FWD**: Falling Weight Deflectometer
* **RUL**: Remaining Useful Life
* **RHI**: Road Health Index
* **ESAL**: Equivalent Single Axle Load
* **LTPP**: Long-Term Pavement Performance