import warnings
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Helper function to handle empty or invalid inputs
def get_float(prompt, default_val):
    val = input(prompt).strip()
    if not val:
        return default_val
    try:
        return float(val)
    except ValueError:
        print(f"Invalid number. Using default: {default_val}")
        return default_val

# 1. Load the models
print("Loading models...")
model1 = joblib.load("../models/iri_prediction_model.pkl")
model2 = joblib.load("../models/fwd_health_model.pkl")
le_target = joblib.load("../models/fwd_label_encoder.pkl")

# Recreate LabelEncoders for new inputs
le_pav = LabelEncoder()
le_pav.fit(['ACUB', 'ACTB', 'ACATB', 'ACPCC', 'JPCUB'])
le_lane = LabelEncoder()
le_lane.fit(['F1', 'F2', 'F3'])

features1 = ['MRI', 'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND',
             'ANNUAL_ESAL_TREND', 'CUMULATIVE_ESAL', 'IRI_GROWTH_RATE', 'YEAR']

features2 = ['PEAK_DEFL_1', 'PEAK_DEFL_2', 'PEAK_DEFL_3', 'PEAK_DEFL_4', 
             'PEAK_DEFL_5', 'PEAK_DEFL_6', 'PEAK_DEFL_7', 'DROP_LOAD', 
             'DROP_HEIGHT', 'PAVEMENT_FAMILY_ENC', 'LANE_NO_ENC']

print("\n=== ROAD HEALTH INDEX PREDICTOR ===\n")

# --- MODEL 1 INPUTS ---
print("--- IRI + Traffic Inputs (Press Enter to use defaults) ---")
mri = get_float("Current IRI/MRI [Default 0.85]: ", 0.85)
aadtt = get_float("Daily Truck Count AADTT [Default 950]: ", 950.0)
annual_truck_vol = get_float("Annual Truck Volume [Default 346750]: ", 346750.0)
annual_esal = get_float("Annual ESAL [Default 310000]: ", 310000.0)
cumulative_esal = get_float("Cumulative ESAL [Default 1500000]: ", 1500000.0)
iri_growth_rate = get_float("IRI Growth Rate [Default 0.012]: ", 0.012)
year = get_float("Current Year [Default 2026]: ", 2026.0)

# --- MODEL 2 INPUTS ---
print("\n--- FWD Inputs (Press Enter to use defaults) ---")
d1 = get_float("PEAK_DEFL_1 [Default 450]: ", 450.0)
d2 = get_float("PEAK_DEFL_2 [Default 280]: ", 280.0)
d3 = get_float("PEAK_DEFL_3 [Default 210]: ", 210.0)
d4 = get_float("PEAK_DEFL_4 [Default 180]: ", 180.0)
d5 = get_float("PEAK_DEFL_5 [Default 140]: ", 140.0)
d6 = get_float("PEAK_DEFL_6 [Default 110]: ", 110.0)
d7 = get_float("PEAK_DEFL_7 [Default 70]: ", 70.0)
drop_load = get_float("DROP_LOAD [Default 710]: ", 710.0)
drop_height = get_float("DROP_HEIGHT [Default 4]: ", 4.0)

print("\nPAVEMENT_FAMILY options: ACUB, ACTB, ACATB, ACPCC, JPCUB")
pav_family = input("PAVEMENT_FAMILY [Default ACUB]: ").strip().upper() or "ACUB"
print("LANE_NO options: F1, F2, F3")
lane_no = input("LANE_NO [Default F3]: ").strip().upper() or "F3"

# --- MODEL 1 PREDICTION ---
input1 = pd.DataFrame([[mri, aadtt, annual_truck_vol, annual_esal,
                         cumulative_esal, iri_growth_rate, year]],
                       columns=features1)

predicted_iri = model1.predict(input1)[0]
iri_change = predicted_iri - mri

if iri_change > 0:
    rul = min((2.5 - mri) / iri_change, 50)
    rul = max(rul, 0)
else:
    rul = 50
iri_score = (rul / 50) * 100

# --- MODEL 2 PREDICTION ---
try:
    pav_enc = le_pav.transform([pav_family])[0]
except:
    pav_enc = 0
try:
    lane_enc = le_lane.transform([lane_no])[0]
except:
    lane_enc = 0

input2 = pd.DataFrame([[d1, d2, d3, d4, d5, d6, d7,
                         drop_load, drop_height, pav_enc, lane_enc]],
                       columns=features2)

health_enc = model2.predict(input2)[0]
health = le_target.inverse_transform([health_enc])[0]
fwd_score = {'Good': 100, 'Fair': 60, 'Poor': 20}[health]

# --- RHI ---
rhi = (iri_score * 0.5) + (fwd_score * 0.5)
if rhi >= 75:
    condition = '🟢 Good'
elif rhi >= 50:
    condition = '🟠 Fair'
else:
    condition = '🔴 Poor'

# --- RESULTS ---
print("\n========== RESULTS ==========")
print(f"Predicted Future IRI  : {predicted_iri:.4f} m/km")
print(f"Remaining Useful Life : {rul:.1f} years")
print(f"IRI Score             : {iri_score:.2f} / 100")
print(f"FWD Health            : {health}")
print(f"FWD Score             : {fwd_score} / 100")
print(f"RHI Score             : {rhi:.2f} / 100")
print(f"Road Condition        : {condition}")
print("==============================")