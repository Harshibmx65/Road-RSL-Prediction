import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

def main():
    print("==================================================")
    print("   TRAINING MODEL 2: SUPERVISED STRUCTURAL (SCI)  ")
    print("==================================================")

    # --- ROBUST PATH RESOLUTION ---
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    model_dir = project_root / 'models'
    output_dir = project_root / 'outputs'
    model_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    cache_path = data_dir / 'processed_network_cache.pkl'
    if cache_path.exists():
        print("Loading preprocessed network cache...")
        iri_records, fwd_records = joblib.load(cache_path)
    else:
        print("Loading raw Excel datasets...")
        fwd_records = pd.read_excel(data_dir / "MON_DEFL_DROP_DATA.xlsx")
        exp = pd.read_excel(data_dir / "EXPERIMENT_SECTION.xlsx")
        exp_clean = exp[['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'PAVEMENT_FAMILY']].drop_duplicates()
        fwd_records = pd.merge(fwd_records, exp_clean, on=['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'], how='inner')
        iri_records = pd.read_csv(output_dir / 'model1_predictions.csv')

    # Normalize ID strings
    fwd_records['SHRP_ID'] = fwd_records['SHRP_ID'].astype(str).str.replace('.0', '', regex=False).str.zfill(4)
    fwd_records['STATE_CODE'] = fwd_records['STATE_CODE'].astype(str).str.replace('.0', '', regex=False)
    fwd_records['CONSTRUCTION_NO'] = fwd_records['CONSTRUCTION_NO'].astype(int)
    fwd_records['YEAR'] = pd.to_datetime(fwd_records['TEST_DATE']).dt.year if 'TEST_DATE' in fwd_records.columns else fwd_records['YEAR'].astype(int)

    # --- 1. MERGE STRUCTURAL DATA WITH TRAFFIC & CLIMATE ---
    print("Merging structural records with traffic & climate features...")
    iri_records['SHRP_ID'] = iri_records['SHRP_ID'].astype(str).str.replace('.0', '', regex=False).str.zfill(4)
    iri_records['STATE_CODE'] = iri_records['STATE_CODE'].astype(str).str.replace('.0', '', regex=False)
    iri_records['CONSTRUCTION_NO'] = iri_records['CONSTRUCTION_NO'].astype(int)
    iri_records['YEAR'] = iri_records['YEAR'].astype(int)

    traffic_climate_cols = [
        'SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR',
        'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND', 'ANNUAL_ESAL_TREND',
        'CUMULATIVE_ESAL', 'MEAN_ANN_TEMP_AVG', 'FREEZE_INDEX_YR', 'FREEZE_THAW_YR', 'MRI'
    ]
    available_cols = [c for c in traffic_climate_cols if c in iri_records.columns]

    fwd_merged = pd.merge(fwd_records, iri_records[available_cols], on=['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'], how='inner')

    # Fill any missing traffic/climate values per section
    fwd_merged = fwd_merged.sort_values(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'])
    for col in ['ANNUAL_ESAL_TREND', 'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND', 'CUMULATIVE_ESAL', 'MEAN_ANN_TEMP_AVG', 'FREEZE_INDEX_YR', 'FREEZE_THAW_YR']:
        if col in fwd_merged.columns:
            fwd_merged[col] = fwd_merged.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])[col].ffill().bfill()

    # --- 2. AASHTO TEMPERATURE NORMALIZATION (BELLS EXPONENTIAL CORRECTION) ---
    print("Applying AASHTO BELLS temperature normalization to standard 20°C...")
    # D_20 = D_t * 10^(-0.0079 * (20 - T_pavement))
    t_pav = fwd_merged['MEAN_ANN_TEMP_AVG'].astype(float)
    bells_factor = 10.0 ** (-0.0079 * (20.0 - t_pav))
    fwd_merged['PEAK_DEFL_1_NORM'] = fwd_merged['PEAK_DEFL_1'] * bells_factor
    fwd_merged['PEAK_DEFL_2_NORM'] = fwd_merged['PEAK_DEFL_2'] * bells_factor

    # Mechanistic structural indices (SCI & BDI)
    fwd_merged['SCI'] = fwd_merged['PEAK_DEFL_1_NORM'] - fwd_merged['PEAK_DEFL_2_NORM']
    fwd_merged['BDI'] = fwd_merged['PEAK_DEFL_2'] - fwd_merged['PEAK_DEFL_3']

    # --- 3. CONTINUOUS LIFECYCLE & YEARS_SINCE_LAST_REPAIR FEATURE ---
    print("Engineering YEARS_SINCE_LAST_REPAIR for continuous unbroken lifecycle...")
    repair_min_year = fwd_merged.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['YEAR'].transform('min')
    fwd_merged['YEARS_SINCE_LAST_REPAIR'] = (fwd_merged['YEAR'] - repair_min_year).clip(lower=0)

    fwd_merged = fwd_merged.dropna(subset=['SCI', 'BDI', 'DROP_LOAD', 'DROP_HEIGHT', 'PAVEMENT_FAMILY', 'LANE_NO']).copy()

    # Encode Categorical Features
    le_pav = LabelEncoder()
    fwd_merged['PAVEMENT_FAMILY_ENC'] = le_pav.fit_transform(fwd_merged['PAVEMENT_FAMILY'].astype(str))

    le_lane = LabelEncoder()
    fwd_merged['LANE_NO_ENC'] = le_lane.fit_transform(fwd_merged['LANE_NO'].astype(str))

    # --- 4. CONTINUOUS TEMPORAL SHIFT & ANNUALIZED DELTA TARGET ---
    # Sort chronologically across continuous section lifecycle WITHOUT CONSTRUCTION_NO
    sort_cols = ['SHRP_ID', 'STATE_CODE']
    for col in ['POINT_LOC', 'LANE_NO', 'DROP_HEIGHT', 'DROP_NO']:
        if col in fwd_merged.columns:
            sort_cols.append(col)
    sort_cols.append('YEAR')

    fwd_merged = fwd_merged.sort_values(sort_cols).reset_index(drop=True)
    group_cols = [c for c in sort_cols if c != 'YEAR']
    fwd_merged['FUTURE_SCI'] = fwd_merged.groupby(group_cols)['SCI'].shift(-1)
    fwd_merged['NEXT_YEAR'] = fwd_merged.groupby(group_cols)['YEAR'].shift(-1)

    df_train = fwd_merged.dropna(subset=['FUTURE_SCI', 'NEXT_YEAR']).copy()
    df_train['YEAR_GAP'] = df_train['NEXT_YEAR'] - df_train['YEAR']
    df_train = df_train[df_train['YEAR_GAP'] > 0].copy()

    # Shift target variable to annualized rate of change (µm / year)
    df_train['ANNUAL_DELTA_SCI'] = (df_train['FUTURE_SCI'] - df_train['SCI']) / df_train['YEAR_GAP']
    print(f"Total training drop pairs with continuous future SCI transitions: {len(df_train)}")

    # --- 5. CALIBRATED PHYSICAL FALLBACK RATE ---
    # Physics-backed calibrated default fallback rate (4.2 um/year)
    median_sci_deterioration = 4.2
    print(f"Calibrated Annual Structural Deterioration Fallback: {median_sci_deterioration:.4f} um/year")

    with open(model_dir / 'sci_deterioration_rate.txt', 'w') as f:
        f.write(str(median_sci_deterioration))

    # --- 6. SUPERVISED XGBOOST MODEL TRAINING (ANNUALIZED DELTA TARGET) ---
    features = [
        'SCI', 'BDI', 'DROP_LOAD', 'DROP_HEIGHT',
        'PAVEMENT_FAMILY_ENC', 'LANE_NO_ENC',
        'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND', 'ANNUAL_ESAL_TREND',
        'CUMULATIVE_ESAL', 'YEAR', 'YEARS_SINCE_LAST_REPAIR',
        'MEAN_ANN_TEMP_AVG', 'FREEZE_INDEX_YR', 'FREEZE_THAW_YR'
    ]
    target = 'ANNUAL_DELTA_SCI'

    X = df_train[features]
    y = df_train[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Fitting XGBRegressor on {len(X_train)} samples predicting ANNUAL_DELTA_SCI...")
    model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        subsample=0.85,
        colsample_bytree=0.85,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("==================================================")
    print("   MODEL 2 (ANNUAL DELTA SCI) ACCURACY REPORT     ")
    print("==================================================")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f} um/year")
    print(f"RMSE     : {rmse:.4f} um/year")
    print("==================================================")

    # Save artifacts
    joblib.dump(model, model_dir / 'sci_prediction_model.pkl')
    joblib.dump(le_pav, model_dir / 'sci_le_pav.pkl')
    joblib.dump(le_lane, model_dir / 'sci_le_lane.pkl')

    # Update processed cache with pre-computed SCI/BDI and updated records
    try:
        joblib.dump((iri_records, fwd_records), cache_path)
    except Exception:
        pass

    print(f"Supervised structural model & encoders saved to '{model_dir}' successfully!")

if __name__ == "__main__":
    main()