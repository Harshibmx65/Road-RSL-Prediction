import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

# --- ROBUST PATH RESOLUTION ---
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / 'data'
model_dir = project_root / 'models'
output_dir = project_root / 'outputs'
model_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

print("Loading raw datasets...")
# Load datasets
iri = pd.read_excel(data_dir / "MON_HSS_PROFILE_SECTION.xlsx")
trf1 = pd.read_excel(data_dir / "TRF_TREND_1.xlsx")
trf2 = pd.read_excel(data_dir / "TRF_TREND.xlsx")
clim = pd.read_excel(data_dir / "CLM_VWS_TEMP_ANNUAL.xlsx")

# --- DATA CLEANING & MERGING ---
print("Cleaning and merging data...")
# Convert date and extract year
iri['VISIT_DATE'] = pd.to_datetime(iri['VISIT_DATE'])
iri['YEAR'] = iri['VISIT_DATE'].dt.year

# Average multiple runs per section per year
iri_clean = iri.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'])['MRI'].mean().reset_index()

# Clean traffic files
trf1['YEAR'] = trf1['YEAR'].astype(int)
trf2['YEAR'] = trf2['YEAR'].astype(int)

# Merge both traffic files (outer join to prevent dropping years with partial data)
trf_merged = pd.merge(
    trf1[['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR', 
          'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND']],
    trf2[['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR', 
          'ANNUAL_ESAL_TREND']],
    on=['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'],
    how='outer'
)

# Merge with IRI
df = pd.merge(
    iri_clean,
    trf_merged,
    on=['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'],
    how='inner'
)

# Sort and Forward-Fill missing traffic data per section to retain maximum training rows
df = df.sort_values(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'])
df['ANNUAL_ESAL_TREND'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['ANNUAL_ESAL_TREND'].ffill()
df['AADTT_ALL_TRUCKS_TREND'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['AADTT_ALL_TRUCKS_TREND'].ffill()
df['ANNUAL_TRUCK_VOLUME_TREND'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['ANNUAL_TRUCK_VOLUME_TREND'].ffill()

# Drop rows that still have nulls
df = df.dropna().reset_index(drop=True)

# Merge Climate Data
clim['SHRP_ID'] = clim['SHRP_ID'].astype(str).str.zfill(4)
clim['STATE_CODE'] = clim['STATE_CODE'].astype(int)
clim['YEAR'] = clim['YEAR'].astype(int)

df['SHRP_ID'] = df['SHRP_ID'].astype(str).str.zfill(4)
df['STATE_CODE'] = df['STATE_CODE'].astype(int)
df['YEAR'] = df['YEAR'].astype(int)

df = pd.merge(df, clim, on=['SHRP_ID', 'STATE_CODE', 'YEAR'], how='inner')

# --- FEATURE ENGINEERING ---
df = df.sort_values(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR']).reset_index(drop=True)

# Cumulative ESAL (total damage load experienced so far)
df['CUMULATIVE_ESAL'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['ANNUAL_ESAL_TREND'].cumsum()

# Target: IRI at next measurement
df['FUTURE_IRI'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['MRI'].shift(-1)

# Keep a copy of the full data before dropping null future IRIs (we need all rows for forecasting)
df_full = df.copy()

# Drop rows without a known future IRI for model training
df_train = df.dropna(subset=['FUTURE_IRI']).copy()

# --- CALCULATE PROFESSIONAL FALLBACK RATE ---
# Find only rows where the road naturally degraded (no maintenance)
natural_degradation = df_train[df_train['FUTURE_IRI'] > df_train['MRI']]

# Calculate the median annual increase in IRI
median_annual_deterioration = float((natural_degradation['FUTURE_IRI'] - natural_degradation['MRI']).median())

print(f"Data-Driven Median Annual Deterioration: {median_annual_deterioration:.4f} m/km/year")

# Save this constant so it can be used in your dashboard and scripts
with open(model_dir / 'deterioration_rate.txt', 'w') as f:
    f.write(str(median_annual_deterioration))

# --- MODEL TRAINING ---
print("Training Model 1 (Surface & Climate)...")
features = ['MRI', 'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND',
            'ANNUAL_ESAL_TREND', 'CUMULATIVE_ESAL', 'YEAR', 
            'MEAN_ANN_TEMP_AVG', 'FREEZE_INDEX_YR', 'FREEZE_THAW_YR']
target = 'FUTURE_IRI'

X = df_train[features]
y = df_train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=6, 
    random_state=42,
    monotone_constraints=(1, 0, 0, 0, 1, 1, 0, 0, 0)
)
model.fit(X_train, y_train)

# Evaluate predictions
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("============================================")
print("       MODEL 1 (CLIMATE) ACCURACY REPORT    ")
print("============================================")
print(f"R² Score  : {r2:.4f}")
print(f"MAE       : {mae:.4f}")
print(f"RMSE      : {rmse:.4f}")
print("============================================")

# Save trained model immediately
joblib.dump(model, model_dir / 'iri_prediction_model.pkl')
print(f"\nModel saved to '{model_dir / 'iri_prediction_model.pkl'}'")

# --- DUAL-RHI SCORING & ITERATIVE FORECASTING ---
FAILURE_THRESHOLD = 2.5
CURRENT_YEAR = 2026

# 1. Calculate the HISTORICAL IRI SCORE (Using the exact year it was measured)
df_full['HISTORICAL_IRI_SCORE'] = (
    ((FAILURE_THRESHOLD - df_full['MRI']) / FAILURE_THRESHOLD) * 100
).clip(0, 100)

# 2. Fast iterative forecasting to 2026
booster = model.get_booster()
from xgboost import DMatrix

def forecast_to_present(row, target_year=CURRENT_YEAR):
    current_mri = float(row['MRI'])
    current_cum_esal = float(row['CUMULATIVE_ESAL'])
    start_year = int(row['YEAR'])
    
    if start_year >= target_year:
        return current_mri

    ann_esal = float(row['ANNUAL_ESAL_TREND'])
    aadtt = float(row['AADTT_ALL_TRUCKS_TREND'])
    truck_vol = float(row['ANNUAL_TRUCK_VOLUME_TREND'])
    temp = float(row['MEAN_ANN_TEMP_AVG'])
    freeze_idx = float(row['FREEZE_INDEX_YR'])
    freeze_thaw = float(row['FREEZE_THAW_YR'])

    for yr in range(start_year, target_year):
        arr = np.array([[current_mri, aadtt, truck_vol, ann_esal, current_cum_esal, yr, temp, freeze_idx, freeze_thaw]], dtype=np.float32)
        dmat = DMatrix(arr, feature_names=features)
        raw_next_mri = float(booster.predict(dmat)[0])
        
        # PROFESSIONAL INFERENCE CLAMP (Data-Driven Heuristic)
        if raw_next_mri <= current_mri:
            # If AI predicts healing or no change, apply the statistical median degradation
            next_mri = current_mri + median_annual_deterioration
        else:
            # If AI predicts valid degradation, trust the AI
            next_mri = raw_next_mri
            
        # Update state for the next loop iteration
        current_mri = next_mri
        current_cum_esal += ann_esal
        
    return current_mri

# 3. Apply the forecast to the entire dataset
print(f"Forecasting all historical sections to the current year ({CURRENT_YEAR})...")
df_full['CURRENT_ESTIMATED_IRI'] = df_full.apply(forecast_to_present, axis=1)

# 4. Calculate the CURRENT IRI SCORE
df_full['CURRENT_IRI_SCORE'] = (
    ((FAILURE_THRESHOLD - df_full['CURRENT_ESTIMATED_IRI']) / FAILURE_THRESHOLD) * 100
).clip(0, 100)

print("\nSample Output:")
print(df_full[['SHRP_ID', 'YEAR', 'MRI', 'HISTORICAL_IRI_SCORE', 'CURRENT_ESTIMATED_IRI', 'CURRENT_IRI_SCORE']].head(10))

df_full.to_csv(output_dir / 'model1_predictions.csv', index=False)
print(f"Predictions saved to '{output_dir / 'model1_predictions.csv'}'")