import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

def main():
    print('\n================================================')
    print('   DUAL-TIMELINE ROAD HEALTH INDEX (RHI) PREDICTOR')
    print('   [Historical Snapshot vs 2026 Present-Day]')
    print('================================================\n')

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
        print("Error: Model files not found in 'models/'. Please run train_model1.py and train_model2.py first.")
        return

    # Load data-driven median annual deterioration rate fallback
    det_rate_file = model_dir / 'deterioration_rate.txt'
    if det_rate_file.exists():
        with open(det_rate_file, 'r') as f:
            median_annual_deterioration = float(f.read().strip())
    else:
        median_annual_deterioration = 0.04

    # Section 1: Road & Historical Surface Measurement Data
    print("--- Section 1: Historical Surface, Traffic & Climate Data ---")
    shrp_id = input('SHRP Section ID (e.g., 0101): ').strip() or 'UNKNOWN'
    measurement_year = int(input('Historical Measurement Year (e.g., 2012 or 2020): '))
    mri = float(input('Measured IRI/MRI at that time (e.g., 0.85): '))
    aadtt = float(input('Daily Truck Count AADTT (e.g., 950): '))
    annual_truck_vol = float(input('Annual Truck Volume (e.g., 346750): '))
    annual_esal = float(input('Annual ESAL (e.g., 310000): '))
    cumulative_esal = float(input('Cumulative ESAL (e.g., 1500000): '))
    
    mean_temp = float(input('Mean Annual Temperature (°C) (e.g., 15.5): '))
    freeze_index = float(input('Annual Freeze Index (e.g., 10): '))
    freeze_thaw = float(input('Annual Freeze-Thaw Cycles (e.g., 45): '))

    features1 = [
        'MRI', 'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND', 
        'ANNUAL_ESAL_TREND', 'CUMULATIVE_ESAL', 'YEAR',
        'MEAN_ANN_TEMP_AVG', 'FREEZE_INDEX_YR', 'FREEZE_THAW_YR'
    ]

    FAILURE_LIMIT = 2.5
    historical_iri_score = float(np.clip(((FAILURE_LIMIT - mri) / FAILURE_LIMIT) * 100, 0, 100))

    # Section 2: Historical FWD Structural Data
    print("\n--- Section 2: Structural Data (Collected at Measurement Time) ---")
    has_fwd_input = input('Do you have FWD Deflection data for this measurement? (y/n): ').strip().lower()
    has_fwd = has_fwd_input == 'y'

    fwd_score = None
    fwd_health = None
    fwd_display = "N/A (Dynamic Fallback Engaged)"

    if has_fwd:
        try:
            deflections = [float(input(f'PEAK_DEFL_{index} (e.g. {value}): '))
                           for index, value in enumerate([450, 280, 210, 180, 140, 110, 70], start=1)]
            drop_load = float(input('DROP_LOAD (e.g. 710): '))
            drop_height = float(input('DROP_HEIGHT (e.g. 4): '))
            
            print(f"Valid Pavement Families: {list(le_pav.classes_)}")
            pav_family = input('PAVEMENT_FAMILY (e.g., ACTB / ACUB): ').strip()
            print(f"Valid Lane Types: {list(le_lane.classes_)}")
            lane_no = input('LANE_NO (e.g., F1 / F3): ').strip()

            features2 = ['PEAK_DEFL_1', 'PEAK_DEFL_2', 'PEAK_DEFL_3', 'PEAK_DEFL_4', 
                         'PEAK_DEFL_5', 'PEAK_DEFL_6', 'PEAK_DEFL_7', 'DROP_LOAD', 
                         'DROP_HEIGHT', 'PAVEMENT_FAMILY_ENC', 'LANE_NO_ENC']
            
            fwd_input = pd.DataFrame([[
                *deflections, drop_load, drop_height,
                le_pav.transform([pav_family])[0],
                le_lane.transform([lane_no])[0]
            ]], columns=features2)
            
            fwd_scaled = scaler_fwd.transform(fwd_input)
            cluster = kmeans_fwd.predict(fwd_scaled)[0]
            fwd_health = health_mapping[cluster]
            
            reverse_mapping = {v: k for k, v in health_mapping.items()}
            good_cluster_idx = reverse_mapping['Good']
            poor_cluster_idx = reverse_mapping['Poor']
            
            distances = kmeans_fwd.transform(fwd_scaled)
            dist_to_good = float(distances[0, good_cluster_idx])
            dist_to_poor = float(distances[0, poor_cluster_idx])
            
            fwd_score = float((dist_to_poor / (dist_to_good + dist_to_poor)) * 100)
            fwd_display = f"{fwd_health} ({fwd_score:.2f}/100)"
        except Exception as e:
            print(f"\nWarning: Could not process FWD input ({e}). Falling back to IRI only.")
            has_fwd = False

    # 1. Historical Score Calculation (Synchronized snapshot at measurement year)
    if has_fwd and fwd_score is not None:
        historical_rhi = float((historical_iri_score * 0.50) + (fwd_score * 0.50))
    else:
        historical_rhi = float(historical_iri_score)

    historical_condition = 'Good' if historical_rhi >= 75 else 'Fair' if historical_rhi >= 50 else 'Poor'

    # 2. AI "Fast-Forward" Simulation to Present Day (2026)
    current_year = 2026
    target_present_year = current_year
    current_mri = mri
    current_cum_esal = cumulative_esal

    if measurement_year < current_year:
        print(f"\nAI Fast-Forwarding deterioration from {measurement_year} to {current_year}...")
        for yr in range(measurement_year, current_year):
            step_input = pd.DataFrame([[
                current_mri, aadtt, annual_truck_vol, annual_esal, 
                current_cum_esal, yr, mean_temp, freeze_index, freeze_thaw
            ]], columns=features1)
            
            raw_next_mri = float(model1.predict(step_input)[0])
            
            # PROFESSIONAL INFERENCE CLAMP (Data-Driven Heuristic)
            if raw_next_mri <= current_mri:
                # If AI predicts healing or no change, apply the statistical median degradation
                next_mri = current_mri + median_annual_deterioration
            else:
                # If AI predicts valid degradation, trust the AI
                next_mri = raw_next_mri
                
            # Update state for the next loop iteration
            current_mri = next_mri
            current_cum_esal += annual_esal
    else:
        current_mri = mri

    # 3. Present Day Score (Dual Timeline Evaluation)
    current_iri_score = max(0.0, min(100.0, ((FAILURE_LIMIT - current_mri) / FAILURE_LIMIT) * 100.0))

    # THE FIX: Check if the data is already from the current year
    if measurement_year == current_year:
        # FWD data is fresh! Do not trigger the fallback. 
        # The Present Day RHI is exactly equal to the Historical RHI.
        current_rhi = historical_rhi 
        structural_policy = "Concurrent FWD data included."
    else:
        # Data is from the past. Trigger the fallback to surface-only.
        current_rhi = current_iri_score 
        structural_policy = "Historic FWD excluded (requires physical re-survey)."

    current_condition = 'Good' if current_rhi >= 75 else 'Fair' if current_rhi >= 50 else 'Poor'

    # --- DISPLAY DUAL REPORT ---
    print(f"\n{'='*60}")
    print(f" ROAD HEALTH INDEX (RHI) DUAL-TIMELINE REPORT")
    print(f" Section: {shrp_id}")
    print(f"{'='*60}")
    print(f"1. HISTORICAL SNAPSHOT ({measurement_year}):")
    print(f"   - Measured IRI        : {mri:.3f} m/km")
    print(f"   - Historical IRI Score: {historical_iri_score:.2f} / 100")
    print(f"   - Structural Health   : {fwd_display}")
    print(f"   - Historical RHI Score: {historical_rhi:.2f} / 100 ({historical_condition})")
    print(f"{'-'*60}")
    print(f"2. PRESENT DAY ESTIMATION ({current_year}):")
    print(f"   - AI Simulated Years  : {max(0, current_year - measurement_year)} years")
    print(f"   - Estimated 2026 IRI  : {current_mri:.3f} m/km (Change: {current_mri - mri:+.3f} m/km)")
    print(f"   - Present Day RHI     : {current_rhi:.2f} / 100 ({current_condition})")
    print(f"   - Structural Policy   : {structural_policy}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()