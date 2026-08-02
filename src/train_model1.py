import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

print("Loading data for Model 1...")
iri = pd.read_excel("../data/MON_HSS_PROFILE_SECTION.xlsx")
trf1 = pd.read_excel("../data/TRF_TREND_1.xlsx")
trf2 = pd.read_excel("../data/TRF_TREND.xlsx")

print("Processing data...")
iri['VISIT_DATE'] = pd.to_datetime(iri['VISIT_DATE'])
iri['YEAR'] = iri['VISIT_DATE'].dt.year
iri = iri.drop(columns=['IRI_CENTER_LANE', 'VISIT_DATE', 'VISIT_NO', 'RUN_NUMBER'])
iri_clean = iri.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'])['MRI'].mean().reset_index()

trf1['YEAR'] = trf1['YEAR'].astype(int)
trf2['YEAR'] = trf2['YEAR'].astype(int)

trf_merged = pd.merge(
    trf1[['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR', 'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND']],
    trf2[['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR', 'ANNUAL_ESAL_TREND']],
    on=['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'], how='inner'
)

df = pd.merge(iri_clean, trf_merged, on=['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR'], how='inner').dropna()
df = df.sort_values(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'YEAR']).reset_index(drop=True)

df['CUMULATIVE_ESAL'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['ANNUAL_ESAL_TREND'].cumsum()

iri_slope = {}
for (shrp, state, cn), group in df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO']):
    if len(group) >= 2:
        slope = np.polyfit(group['YEAR'], group['MRI'], 1)[0]
        iri_slope[(shrp, state, cn)] = slope

df['IRI_GROWTH_RATE'] = df.apply(lambda row: iri_slope.get((row['SHRP_ID'], row['STATE_CODE'], row['CONSTRUCTION_NO']), np.nan), axis=1)
df['FUTURE_IRI'] = df.groupby(['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'])['MRI'].shift(-1)
df = df.dropna(subset=['FUTURE_IRI', 'IRI_GROWTH_RATE'])

features = ['MRI', 'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND', 'ANNUAL_ESAL_TREND', 'CUMULATIVE_ESAL', 'IRI_GROWTH_RATE', 'YEAR']
X = df[features]
y = df['FUTURE_IRI']

print("Training Model 1 (XGBoost Regressor)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, '../models/iri_prediction_model.pkl')
print("Model 1 saved successfully to models/iri_prediction_model.pkl!")