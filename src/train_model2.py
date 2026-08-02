import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

print("Loading data for Model 2...")
fwd = pd.read_excel("../data/MON_DEFL_DROP_DATA.xlsx")
exp = pd.read_excel("../data/EXPERIMENT_SECTION.xlsx")

print("Processing data...")
fwd['SHRP_ID'] = fwd['SHRP_ID'].astype(str).str.zfill(4)
exp_clean = exp[['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'PAVEMENT_FAMILY']].drop_duplicates()
fwd = fwd.drop(columns=['PEAK_DEFL_9', 'NON_DECREASING_DEFL', 'STATE_CODE_EXP', 'LANE_NO_EXP', 'DROP_HEIGHT_EXP', 'DEFL_UNIT_ID'], errors='ignore')

df2 = pd.merge(fwd, exp_clean, on=['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'], how='inner')
df2 = df2.drop(columns=['PEAK_DEFL_8', 'NON_DECREASING_DEFL_EXP'], errors='ignore').dropna()

df2['AVG_DEFL'] = df2[['PEAK_DEFL_1','PEAK_DEFL_2','PEAK_DEFL_3','PEAK_DEFL_4','PEAK_DEFL_5','PEAK_DEFL_6','PEAK_DEFL_7']].mean(axis=1)

def classify_health(defl):
    if defl < 300: return 'Good'
    elif defl <= 500: return 'Fair'
    else: return 'Poor'

df2['HEALTH'] = df2['AVG_DEFL'].apply(classify_health)

le = LabelEncoder()
df2['PAVEMENT_FAMILY_ENC'] = le.fit_transform(df2['PAVEMENT_FAMILY'])
df2['LANE_NO_ENC'] = le.fit_transform(df2['LANE_NO'])

features2 = ['PEAK_DEFL_1', 'PEAK_DEFL_2', 'PEAK_DEFL_3', 'PEAK_DEFL_4', 'PEAK_DEFL_5', 'PEAK_DEFL_6', 'PEAK_DEFL_7', 'DROP_LOAD', 'DROP_HEIGHT', 'PAVEMENT_FAMILY_ENC', 'LANE_NO_ENC']
X2 = df2[features2]
y2 = df2['HEALTH']

le_target = LabelEncoder()
y2_enc = le_target.fit_transform(y2)

print("Training Model 2 (XGBoost Classifier)...")
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2_enc, test_size=0.2, random_state=42, stratify=y2_enc)

model2 = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='mlogloss')
model2.fit(X2_train, y2_train)

joblib.dump(model2, '../models/fwd_health_model.pkl')
joblib.dump(le_target, '../models/fwd_label_encoder.pkl')
print("Model 2 saved successfully to models/fwd_health_model.pkl!")