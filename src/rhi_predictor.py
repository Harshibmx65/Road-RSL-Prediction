import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

def main():
    print('\n=== ROAD HEALTH INDEX PREDICTOR ===\n')

    # --- ROBUST PATH RESOLUTION ---
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models'
    # ------------------------------

    # Load saved models and preprocessors
    try:
        model1 = joblib.load(model_dir / 'iri_prediction_model.pkl')
        kmeans_fwd = joblib.load(model_dir / 'fwd_kmeans_model.pkl')
        scaler_fwd = joblib.load(model_dir / 'fwd_scaler.pkl')
        le_pav = joblib.load(model_dir / 'fwd_le_pav.pkl')
        le_lane = joblib.load(model_dir / 'fwd_le_lane.pkl')
        health_mapping = joblib.load(model_dir / 'fwd_health_mapping.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_model1.py and train_model2.py first.")
        return

    # IRI and traffic inputs
    print("--- Section 1: Surface & Traffic Data ---")
    mri = float(input('Current IRI/MRI (e.g. 0.85): '))
    aadtt = float(input('Daily Truck Count AADTT (e.g. 950): '))
    annual_truck_vol = float(input('Annual Truck Volume (e.g. 346750): '))
    annual_esal = float(input('Annual ESAL (e.g. 310000): '))
    cumulative_esal = float(input('Cumulative ESAL (e.g. 1500000): '))
    year = float(input('Current Year (e.g. 2025): '))

    features1 = ['MRI', 'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND', 
                 'ANNUAL_ESAL_TREND', 'CUMULATIVE_ESAL', 'YEAR']

    iri_input = pd.DataFrame([[mri, aadtt, annual_truck_vol, annual_esal, 
                               cumulative_esal, year]], columns=features1)

    predicted_iri = float(model1.predict(iri_input)[0])
    failure_threshold = 2.5
    iri_score = float(np.clip(
        ((failure_threshold - predicted_iri) / failure_threshold) * 100, 0, 100
    ))

    # FWD inputs with dynamic fallback prompt
    print("\n--- Section 2: Structural Data ---")
    has_fwd = input('Do you have FWD Deflection data for this section? (y/n): ').strip().lower()

    if has_fwd == 'y':
        deflections = [float(input(f'PEAK_DEFL_{index} (e.g. {value}): '))
                       for index, value in enumerate([450, 280, 210, 180, 140, 110, 70], start=1)]
        drop_load = float(input('DROP_LOAD (e.g. 710): '))
        drop_height = float(input('DROP_HEIGHT (e.g. 4): '))
        
        print(f"Valid Pavement Families: {list(le_pav.classes_)}")
        pav_family = input('PAVEMENT_FAMILY: ')
        print(f"Valid Lane Types: {list(le_lane.classes_)}")
        lane_no = input('LANE_NO: ')

        features2 = ['PEAK_DEFL_1', 'PEAK_DEFL_2', 'PEAK_DEFL_3', 'PEAK_DEFL_4', 
                     'PEAK_DEFL_5', 'PEAK_DEFL_6', 'PEAK_DEFL_7', 'DROP_LOAD', 
                     'DROP_HEIGHT', 'PAVEMENT_FAMILY_ENC', 'LANE_NO_ENC']
        
        try:
            fwd_input = pd.DataFrame([[
                *deflections, drop_load, drop_height,
                le_pav.transform([pav_family])[0],
                le_lane.transform([lane_no])[0]
            ]], columns=features2)
            
            # Scale features and predict cluster
            fwd_scaled = scaler_fwd.transform(fwd_input)
            cluster = kmeans_fwd.predict(fwd_scaled)[0]
            
            health = health_mapping[cluster]
            fwd_score = {'Good': 100, 'Fair': 60, 'Poor': 20}[health]
            
            rhi = (iri_score + fwd_score) / 2
            fwd_display = f"{health} ({fwd_score}/100)"
        except ValueError as e:
            print(f"\nError encoding categorical data: {e}")
            print("Falling back to IRI-only RHI score due to invalid input.")
            rhi = iri_score
            fwd_display = "N/A (Invalid Input - Fallback Engaged)"
    else:
        rhi = iri_score
        fwd_display = "N/A (Dynamic Fallback Engaged)"

    condition = 'Good' if rhi >= 75 else 'Fair' if rhi >= 50 else 'Poor'

    print('\n========== RESULTS ==========')
    print(f'Predicted Future IRI  : {predicted_iri:.4f} m/km')
    print(f'IRI Score             : {iri_score:.2f} / 100')
    print(f'FWD Structural Health : {fwd_display}')
    print(f'Final RHI Score       : {rhi:.2f} / 100')
    print(f'Overall Road Condition: {condition}')
    print('==============================')

if __name__ == "__main__":
    main()