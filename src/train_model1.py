import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

def main():
    print("Loading data for Model 1 (IRI & Climate)...")
    
    # --- ROBUST PATH RESOLUTION ---
    # Gets the directory of this script, then goes up one level to the root
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    model_dir = project_root / 'models'
    model_dir.mkdir(exist_ok=True)
    # ------------------------------

    # Load datasets
    iri = pd.read_excel(data_dir / "MON_HSS_PROFILE_SECTION.xlsx")
    trf1 = pd.read_excel(data_dir / "TRF_TREND_1.xlsx")
    trf2 = pd.read_excel(data_dir / "TRF_TREND.xlsx")
    clim = pd.read_excel(data_dir / "CLM_VWS_TEMP_ANNUAL.xlsx")

    # Clean IRI
    iri['VISIT_DATE'] = pd.to_datetime(iri['VISIT_DATE'])
    iri['YEAR'] = iri['VISIT_DATE'].dt.year
    iri = iri.drop(columns=['IRI_CENTER_LANE', 'VISIT_DATE', 'VISIT_NO', 'RUN_NUMBER'], errors='ignore')
    iri_clean = iri.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'])['MRI'].mean().reset_index()

    # Clean Traffic
    trf1['YEAR'] = trf1['YEAR'].astype(int)
    trf2['YEAR'] = trf2['YEAR'].astype(int)

    # Merge Traffic (Outer join to prevent dropping years with partial data)
    trf_merged = pd.merge(
        trf1[['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR', 'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND']],
        trf2[['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR', 'ANNUAL_ESAL_TREND']],
        on=['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'],
        how='outer'
    )

    # Merge IRI and Traffic
    df = pd.merge(iri_clean, trf_merged, on=['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'], how='inner')
    
    # Forward-Fill Imputation for Traffic
    df = df.sort_values(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'])
    df['ANNUAL_ESAL_TREND'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['ANNUAL_ESAL_TREND'].ffill()
    df['AADTT_ALL_TRUCKS_TREND'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['AADTT_ALL_TRUCKS_TREND'].ffill()
    df['ANNUAL_TRUCK_VOLUME_TREND'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['ANNUAL_TRUCK_VOLUME_TREND'].ffill()
    df = df.dropna().reset_index(drop=True)

    # Clean Climate Data Formatting
    clim['SHRP_ID'] = clim['SHRP_ID'].astype(str).str.zfill(4)
    clim['STATE_CODE'] = clim['STATE_CODE'].astype(int)
    clim['YEAR'] = clim['YEAR'].astype(int)

    df['SHRP_ID'] = df['SHRP_ID'].astype(str).str.zfill(4)
    df['STATE_CODE'] = df['STATE_CODE'].astype(int)
    df['YEAR'] = df['YEAR'].astype(int)

    # Merge with Climate Data
    df = pd.merge(df, clim, on=['SHRP_ID', 'STATE_CODE', 'YEAR'], how='inner')

    # Feature Engineering
    df['CUMULATIVE_ESAL'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['ANNUAL_ESAL_TREND'].cumsum()
    df['FUTURE_IRI'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['MRI'].shift(-1)
    df = df.dropna(subset=['FUTURE_IRI'])

    # Updated Features List including environmental stressors
    features = [
        'MRI', 'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND', 
        'ANNUAL_ESAL_TREND', 'CUMULATIVE_ESAL', 'YEAR', 
        'MEAN_ANN_TEMP_AVG', 'FREEZE_INDEX_YR', 'FREEZE_THAW_YR'
    ]
    target = 'FUTURE_IRI'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost Regressor with Climate Data...")
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model 1 Trained. R²: {r2:.4f}, MAE: {mae:.4f}")

    joblib.dump(model, model_dir / 'iri_prediction_model.pkl')
    print("Model 1 saved to models/iri_prediction_model.pkl")

if __name__ == "__main__":
    main()