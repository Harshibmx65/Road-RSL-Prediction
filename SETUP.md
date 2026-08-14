# 🛠️ Complete Beginner Setup & Installation Manual

> **A step-by-step, zero-to-hero guide to setting up, running, training, and testing the Road Health Index (RHI) & Pavement Remaining Service Life Prediction System on Windows, macOS, and Linux.**

---

## 📑 Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Step-by-Step Installation Guide](#2-step-by-step-installation-guide)
   * [Step 1: Open Terminal & Navigate to Project](#step-1-open-terminal--navigate-to-project)
   * [Step 2: Check Python & Pip Installation](#step-2-check-python--pip-installation)
   * [Step 3: Create a Virtual Environment (`venv`)](#step-3-create-a-virtual-environment-venv)
   * [Step 4: Activate the Virtual Environment](#step-4-activate-the-virtual-environment)
   * [Step 5: Install Project Dependencies](#step-5-install-project-dependencies)
   * [Step 6: Verify Raw Datasets](#step-6-verify-raw-datasets)
   * [Step 7: Train the Machine Learning Models](#step-7-train-the-machine-learning-models)
   * [Step 8: Verify Generated Model Artifacts](#step-8-verify-generated-model-artifacts)
   * [Step 9: Run the Interactive Terminal CLI Predictor](#step-9-run-the-interactive-terminal-cli-predictor)
   * [Step 10: Launch the Web Control Center (FastAPI Dashboard)](#step-10-launch-the-web-control-center-fastapi-dashboard)
   * [Step 11: Explore Interactive REST API Docs (Swagger UI)](#step-11-explore-interactive-rest-api-docs-swagger-ui)
   * [Step 12: Run Jupyter Notebooks & Verification Tests](#step-12-run-jupyter-notebooks--verification-tests)
3. [Understanding the Dependencies (`requirements.txt`)](#3-understanding-the-dependencies-requirementstxt)
4. [Step-by-Step Web Dashboard User Guide](#4-step-by-step-web-dashboard-user-guide)
5. [Troubleshooting & Frequently Asked Questions (FAQ)](#5-troubleshooting--frequently-asked-questions-faq)
6. [Quick Reference Command Cheat Sheet](#6-quick-reference-command-cheat-sheet)

---

## 1. Prerequisites & System Requirements

Before starting, ensure your system meets the following minimal requirements:

| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, macOS (Intel or Apple Silicon), Ubuntu 20.04+ | Windows 11 / macOS / Ubuntu |
| **Python Version** | Python 3.10.x | **Python 3.11.x – 3.13.x** |
| **RAM** | 4 GB | 8 GB or higher |
| **Disk Space** | 500 MB (includes datasets and models) | 1 GB free space |
| **Web Browser** | Any modern browser | Google Chrome, Microsoft Edge, Firefox, Brave |
| **Code Editor (Optional)** | None required (runs in terminal) | VS Code or Cursor |

---

## 2. Step-by-Step Installation Guide

Follow these numbered steps sequentially.

---

### Step 1: Open Terminal & Navigate to Project

Open your favorite terminal:
* **Windows**: Open **PowerShell** or **Command Prompt (cmd)**.
* **macOS**: Open **Terminal** (via Spotlight: `Cmd + Space` $\to$ type `Terminal`).
* **Linux**: Open your shell terminal (`Ctrl + Alt + T`).

Navigate into the root directory of the project:

```powershell
# Windows PowerShell / CMD
cd C:\Users\rajan\Road-RSL-Prediction
```

```bash
# macOS / Linux
cd /path/to/Road-RSL-Prediction
```

---

### Step 2: Check Python & Pip Installation

Verify that Python is installed and available in your system path:

```powershell
python --version
```

* Expected output: `Python 3.10.x`, `Python 3.11.x`, `Python 3.12.x`, or `Python 3.13.x`.
* If `python` is not recognized, try `python3 --version` or `py --version`.

Check that `pip` (Python Package Installer) is present:

```powershell
python -m pip --version
```

> [!NOTE]
> If Python is not installed, download the official installer from [python.org](https://www.python.org/downloads/). During installation on Windows, **make sure to check the box: "Add Python to PATH"**.

---

### Step 3: Create a Virtual Environment (`venv`)

A **Virtual Environment** is an isolated folder on your computer that keeps this project's packages separate from other Python programs. This prevents version conflicts and keeps your system clean.

Run the following command in the project root:

```powershell
python -m venv venv
```

This creates a new folder named `venv/` inside your project containing a dedicated Python executable and library directory.

---

### Step 4: Activate the Virtual Environment

You must **activate** the virtual environment whenever you work on this project.

#### On Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```

> [!IMPORTANT]
> **PowerShell Execution Policy Fix**: If you encounter an error like `cannot be loaded because running scripts is disabled on this system`, run this one-time command in PowerShell and try activating again:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

#### On Windows (Command Prompt `cmd.exe`):
```cmd
venv\Scripts\activate.bat
```

#### On macOS / Linux (Bash or Zsh):
```bash
source venv/bin/activate
```

**How do you know it worked?**
Your terminal prompt will now show `(venv)` at the beginning of the line:
```powershell
(venv) PS C:\Users\rajan\Road-RSL-Prediction>
```

---

### Step 5: Install Project Dependencies

Upgrade `pip` and install all required libraries listed in [`requirements.txt`](file:///c:/Users/rajan/Road-RSL-Prediction/requirements.txt):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This will automatically install:
* `pandas` and `openpyxl` (data processing and Excel reading)
* `numpy` (vectorized numerical calculations)
* `scikit-learn` (StandardScaler, LabelEncoder, KMeans)
* `xgboost` (extreme gradient boosting regression)
* `fastapi` and `uvicorn[standard]` (web server and REST API)
* `joblib` (saving and loading model files)
* `matplotlib` (generating research plots)
* `jupyter` (running interactive analysis notebooks)
* `python-multipart` (handling batch file uploads in the dashboard)

---

### Step 6: Verify Raw Datasets

Make sure all 6 source Excel workbooks exist in the [`data/`](file:///c:/Users/rajan/Road-RSL-Prediction/data/) directory:

```powershell
# Windows PowerShell
Get-ChildItem -Path data\*.xlsx
```

```bash
# macOS / Linux
ls -la data/*.xlsx
```

You should see all 6 files:
1. `CLM_VWS_TEMP_ANNUAL.xlsx` (Climate records)
2. `EXPERIMENT_SECTION.xlsx` (Pavement design metadata)
3. `MON_DEFL_DROP_DATA.xlsx` (FWD sensor drop deflections)
4. `MON_HSS_PROFILE_SECTION.xlsx` (Laser profilometer roughness scans)
5. `TRF_TREND.xlsx` (Annual ESAL damage metrics)
6. `TRF_TREND_1.xlsx` (Truck volume and AADTT counts)

---

### Step 7: Train the Machine Learning Models

The repository comes with pre-trained models, but you can train both models from scratch at any time:

#### 1. Train Model 1 (Surface Roughness & Climate XGBoost Regressor):
```powershell
python src/train_model1.py
```
* **What happens**: The script loads profiler scans, traffic, and temperature records, fills missing temporal trends, trains an XGBoost Regressor ($R^2 \approx 0.70$–$0.85$), and saves [`models/iri_prediction_model.pkl`](file:///c:/Users/rajan/Road-RSL-Prediction/models/iri_prediction_model.pkl).

#### 2. Train Model 2 (FWD Structural Deflection K-Means Clustering):
```powershell
python src/train_model2.py
```
* **What happens**: The script processes 7-sensor deflection drop data, encodes pavement families, scales features, fits a 3-cluster K-Means model, sorts centroids by average deflection to assign Good/Fair/Poor health ratings, and saves 5 artifacts to [`models/`](file:///c:/Users/rajan/Road-RSL-Prediction/models/).

---

### Step 8: Verify Generated Model Artifacts

Check that the [`models/`](file:///c:/Users/rajan/Road-RSL-Prediction/models/) directory contains all 6 required `.pkl` files:

```powershell
# Windows PowerShell
Get-ChildItem -Path models\*.pkl
```

```bash
# macOS / Linux
ls -la models/*.pkl
```

You should see:
* `iri_prediction_model.pkl`
* `fwd_kmeans_model.pkl`
* `fwd_scaler.pkl`
* `fwd_le_pav.pkl`
* `fwd_le_lane.pkl`
* `fwd_health_mapping.pkl`

---

### Step 9: Run the Interactive Terminal CLI Predictor

To test an individual road section directly in your terminal:

```powershell
python src/rhi_predictor.py
```

**Example Interactive Session**:
```text
=== ROAD HEALTH INDEX PREDICTOR ===

--- Section 1: Surface, Traffic & Climate Data ---
Current IRI/MRI (e.g. 0.85): 0.85
Daily Truck Count AADTT (e.g. 950): 950
Annual Truck Volume (e.g. 346750): 346750
Annual ESAL (e.g. 310000): 310000
Cumulative ESAL (e.g. 1500000): 1500000
Current Year (e.g. 2025): 2025
Mean Annual Temperature (C) (e.g. 15.5): 15.5
Annual Freeze Index (e.g. 10): 10
Annual Freeze-Thaw Cycles (e.g. 45): 45

--- Section 2: Structural Data ---
Do you have FWD Deflection data for this section? (y/n): y
PEAK_DEFL_1 (e.g. 450): 450
PEAK_DEFL_2 (e.g. 280): 280
PEAK_DEFL_3 (e.g. 210): 210
PEAK_DEFL_4 (e.g. 180): 180
PEAK_DEFL_5 (e.g. 140): 140
PEAK_DEFL_6 (e.g. 110): 110
PEAK_DEFL_7 (e.g. 70): 70
DROP_LOAD (e.g. 710): 710
DROP_HEIGHT (e.g. 4): 4
Valid Pavement Families: ['ACTB', 'ACUB']
PAVEMENT_FAMILY: ACUB
Valid Lane Types: ['F1', 'F3']
LANE_NO: F3

========== RESULTS ==========
Predicted Future IRI  : 0.9124 m/km
IRI Score             : 63.50 / 100
FWD Structural Health : Fair (60.23/100)
Final RHI Score       : 61.87 / 100
Overall Road Condition: Fair
==============================
```

---

### Step 10: Launch the Web Control Center (FastAPI Dashboard)

Start the production FastAPI server:

```powershell
python -m uvicorn Dashboard.main:app --reload --port 8000
```

* `--reload`: Automatically reloads the server if Python code changes.
* `--port 8000`: Runs the application on port 8000.

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`** (or `http://localhost:8000`)

---

### Step 11: Explore Interactive REST API Docs (Swagger UI)

FastAPI automatically generates interactive OpenAPI documentation.
Open your browser and visit:
👉 **`http://127.0.0.1:8000/docs`**

Here you can:
* Test every REST endpoint (`/api/sections`, `/api/section/{id}`, `/api/predict`, `/api/batch`, `/api/report.csv`).
* Click **"Try it out"**, fill in parameters, and click **"Execute"** to see live JSON responses.

---

### Step 12: Run Jupyter Notebooks & Verification Tests

To inspect the exploratory data analysis, visualizations, and training experiments:

```powershell
jupyter notebook
```

Your browser will open the Jupyter dashboard. You can run:
1. [`notebooks/model1.ipynb`](file:///c:/Users/rajan/Road-RSL-Prediction/notebooks/model1.ipynb) $\to$ Model 1 development & evaluation plots.
2. [`notebooks/model2.ipynb`](file:///c:/Users/rajan/Road-RSL-Prediction/notebooks/model2.ipynb) $\to$ Model 2 K-Means structural clustering.
3. [`notebooks/RHI_Score.ipynb`](file:///c:/Users/rajan/Road-RSL-Prediction/notebooks/RHI_Score.ipynb) $\to$ Full network fusion pipeline & CSV generation.
4. [`testing/test_rhi_score.ipynb`](file:///c:/Users/rajan/Road-RSL-Prediction/testing/test_rhi_score.ipynb) $\to$ Standalone single-section verification script.

---

## 3. Understanding the Dependencies (`requirements.txt`)

Here is an explanation of every library in [`requirements.txt`](file:///c:/Users/rajan/Road-RSL-Prediction/requirements.txt):

| Package | Minimum Version | Why It Is Needed |
| :--- | :--- | :--- |
| **`pandas`** | $\ge 2.0$ | Core data manipulation library used to clean, merge, forward-fill, and group tabular road datasets. |
| **`openpyxl`** | $\ge 3.1$ | Excel workbook engine allowing pandas to read and parse `.xlsx` data files from the LTPP database. |
| **`numpy`** | $\ge 1.24$ | Numerical computing library used for matrix operations, clipping bounds, and Euclidean distance scoring. |
| **`scikit-learn`** | $\ge 1.3$ | Machine learning toolkit providing `KMeans`, `StandardScaler`, `LabelEncoder`, and `train_test_split`. |
| **`matplotlib`** | $\ge 3.7$ | Plotting library used in Jupyter notebooks to render scatter plots, cluster projections, and histograms. |
| **`joblib`** | $\ge 1.3$ | High-efficiency object serialization library used to save and load `.pkl` trained model artifacts. |
| **`jupyter`** | $\ge 1.0$ | Interactive computational environment for executing `.ipynb` research and validation notebooks. |
| **`fastapi`** | $\ge 0.115$ | High-performance asynchronous web framework powering the backend REST API endpoints. |
| **`uvicorn[standard]`** | $\ge 0.30$ | Lightning-fast ASGI web server implementation running the FastAPI application. |
| **`xgboost`** | $\ge 2.0$ | State-of-the-art gradient boosted decision tree framework used to train the Model 1 regression engine. |
| **`python-multipart`** | $\ge 0.0.9$ | Middleware enabling FastAPI to receive and process uploaded `.csv` and `.xlsx` batch assessment files. |

---

## 4. Step-by-Step Web Dashboard User Guide

Once you open **`http://127.0.0.1:8000`** in your browser, here is how to use every feature:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🛣️ ROAD HEALTH INDEX | CONTROL CENTER                                 502 Sections Monitored│
├──────────────────────────────┬──────────────────────────────────────────────────────────────┤
│ SECTION LOOKUP               │ FINAL ROAD HEALTH INDEX           COMPONENT BREAKDOWN        │
│ [Search: 0101            ]   │       ╭─────────╮                 ┌───────────────────────┐  │
│ ┌──────────────────────────┐ │      │   61.8   │                 │ [██████] IRI: 63.5    │  │
│ │ SHRP 0101 · State 1      │ │       ╰─────────╯                 │ [█████ ] FWD: 60.0    │  │
│ └──────────────────────────┘ │       FAIR CONDITION              └───────────────────────┘  │
│                              ├──────────────────────────────────────────────────────────────┤
│ SURFACE & TRAFFIC INPUTS     │ DETERIORATION TIMELINE            NETWORK CONDITION          │
│ IRI / MRI:     [0.85     ]   │ [Line Chart: Historical + 10-Yr]  [Doughnut: Good/Fair/Poor] │
│ Daily Trucks:  [950      ]   ├──────────────────────────────────────────────────────────────┤
│ Cumulative:    [1500000  ]   │ BATCH ASSESSMENT (Upload CSV/XLSX for up to 50 roads)        │
│ Mean Temp:     [12.5     ]   ├──────────────────────────────────────────────────────────────┤
│ [✓] FWD Data Available       │ REPORTING: [Download CSV] [Download PDF]                     │
│ [Update Live Prediction]     ├──────────────────────────────────────────────────────────────┤
│                              │ TEST RHI PREDICTOR: [Run Sample Test]                        │
└──────────────────────────────┴──────────────────────────────────────────────────────────────┘
```

### 1. Section Search & Historical Loading
* Click in the **"Find a road segment"** box in the left sidebar.
* Type a 4-digit SHRP ID (e.g. `0101`, `0102`, `0103`) or a State code.
* Click on a result.
* **What happens**: The system automatically queries `/api/section/{id}`, populates the input form with that road's latest historical parameters, updates the 10-year timeline chart, and renders the live RHI gauge.

### 2. Live What-If Scenario Simulation
* Adjust any input parameter (e.g., increase `Daily truck count` to test heavy freight traffic growth, or increase `Current IRI` to simulate surface aging).
* Click **"Update live prediction"**.
* **What happens**: The dashboard immediately calculates the new predicted IRI, updates the SVG gauge meter, highlights top contributing factors via SHAP explanations, and recalculates the 10-year deterioration forecast.

### 3. Testing Dynamic Fallback (Missing FWD Sensors)
* In the left sidebar, uncheck the **"FWD structural data available"** toggle switch.
* Click **"Update live prediction"**.
* **What happens**: The FWD input section hides, an amber badge appears stating *"Fallback active · RHI based 100% on IRI"*, and the RHI score smoothly computes using pure surface and climate data.

### 4. Batch Road Assessment
* Scroll down to the **Batch Assessment** section.
* Click **"Choose File"** and select a `.csv` or `.xlsx` containing multiple road segments (with headers: `MRI`, `AADTT_ALL_TRUCKS_TREND`, `ANNUAL_TRUCK_VOLUME_TREND`, `ANNUAL_ESAL_TREND`, `CUMULATIVE_ESAL`, `YEAR`).
* Click **"Upload & download results"**.
* **What happens**: The server processes all rows in parallel and immediately streams a downloaded `batch-rhi-results.csv` file with scores for every row.

### 5. Exporting Inspection Reports
* Click **"Download CSV"** to get a comma-separated values report of the active scenario.
* Click **"Download PDF"** to generate an executive, styled assessment summary card suitable for civil engineering presentations.

### 6. Running the Sample Test Verification
* Scroll to the **"Test RHI Predictor"** section at the bottom.
* Click **"Run sample test"**.
* **What happens**: The dashboard runs the exact benchmark profile from the test notebook and renders a verification card with Surface Score, Structural Score, and RHI.

---

## 5. Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: PowerShell error: `File Activate.ps1 cannot be loaded because running scripts is disabled`
* **Cause**: Windows PowerShell security policy restricts unsigned script execution by default.
* **Fix**: Run this command in your PowerShell window to allow script execution for the current terminal session:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1
  ```

---

### Q2: Error when starting server: `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`
* **Cause**: Another application or an older instance of uvicorn is already using port 8000.
* **Fix**: Either close the other process, or specify a different port:
  ```powershell
  python -m uvicorn Dashboard.main:app --reload --port 8080
  ```
  Then open `http://127.0.0.1:8080` in your browser.

---

### Q3: `FileNotFoundError: Missing model artifacts: iri_prediction_model.pkl... Train the models first.`
* **Cause**: The pre-trained model pickle files were deleted or have not been trained yet.
* **Fix**: Run the training scripts from the project root:
  ```powershell
  python src/train_model1.py
  python src/train_model2.py
  ```

---

### Q4: `ImportError: Missing optional dependency 'openpyxl'`
* **Cause**: `openpyxl` was not installed in your active virtual environment.
* **Fix**: Ensure your virtual environment is activated (`(venv)` shown in prompt), then install it:
  ```powershell
  python -m pip install openpyxl
  ```

---

### Q5: `ValueError: Unsupported category: pavement=...` when running predictions
* **Cause**: The entered `PAVEMENT_FAMILY` or `LANE_NO` does not match the LTPP training categories.
* **Fix**: Use only supported values:
  * **Pavement Families**: `ACTB` (Asphalt Concrete over Treated Base) or `ACUB` (Asphalt Concrete over Untreated Base).
  * **Lane Types**: `F1` (Outer Traffic Lane) or `F3` (Inner Passing Lane).

---

### Q6: How do I stop the FastAPI server?
* **Answer**: Press **`Ctrl + C`** in the terminal where uvicorn is running.

---

### Q7: How do I deactivate the virtual environment when finished?
* **Answer**: Simply type:
  ```powershell
  deactivate
  ```

---

## 6. Quick Reference Command Cheat Sheet

| Task | PowerShell / Windows Command | macOS / Linux Command |
| :--- | :--- | :--- |
| **Navigate to project** | `cd C:\Users\rajan\Road-RSL-Prediction` | `cd ~/Road-RSL-Prediction` |
| **Create virtual env** | `python -m venv venv` | `python3 -m venv venv` |
| **Activate virtual env** | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| **Install dependencies** | `python -m pip install -r requirements.txt` | `python3 -m pip install -r requirements.txt` |
| **Train Model 1 (IRI)** | `python src/train_model1.py` | `python3 src/train_model1.py` |
| **Train Model 2 (FWD)** | `python src/train_model2.py` | `python3 src/train_model2.py` |
| **Run CLI predictor** | `python src/rhi_predictor.py` | `python3 src/rhi_predictor.py` |
| **Start Web Dashboard** | `python -m uvicorn Dashboard.main:app --reload --port 8000` | `python3 -m uvicorn Dashboard.main:app --reload --port 8000` |
| **Launch Jupyter** | `jupyter notebook` | `jupyter notebook` |
| **Deactivate env** | `deactivate` | `deactivate` |

---

### 📖 Looking for Code Documentation?
To understand the machine learning algorithms, mathematical formulas, system architecture, and exhaustive function-by-function catalog, see **[`README.md`](file:///c:/Users/rajan/Road-RSL-Prediction/README.md)**.
