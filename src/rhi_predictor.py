import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

IRI_FAILURE_LIMIT = 2.5
SCI_FAILURE_LIMIT = 200.0

def score_iri(iri_val: float) -> float:
    return float(np.clip(((IRI_FAILURE_LIMIT - iri_val) / IRI_FAILURE_LIMIT) * 100.0, 0.0, 100.0))

def score_sci(sci_val: float) -> float:
    return float(np.clip(((SCI_FAILURE_LIMIT - sci_val) / SCI_FAILURE_LIMIT) * 100.0, 0.0, 100.0))

def condition_label(score: float) -> str:
    if score >= 75.0:
        return 'Good'
    if score >= 50.0:
        return 'Fair'
    return 'Poor'

def main():
    print('\n================================================')
    print('   DUAL-TIMELINE ROAD HEALTH INDEX (RHI) PREDICTOR')
    print('   [Supervised Synchronized Surface + Structural AI]')
    print('================================================\n')

    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models'

    try:
        model_iri = joblib.load(model_dir / 'iri_prediction_model.pkl')
        model_sci = joblib.load(model_dir / 'sci_prediction_model.pkl')
        le_pav = joblib.load(model_dir / 'sci_le_pav.pkl')
        le_lane = joblib.load(model_dir / 'sci_le_lane.pkl')
    except FileNotFoundError as e:
        print(f"Error: Model files not found in 'models/'. Please run train_model1.py and train_model2.py first ({e}).")
        return

    det_rate_file = model_dir / 'deterioration_rate.txt'
    iri_det_rate = 0.04
    if det_rate_file.exists():
        with open(det_rate_file, 'r') as f:
            iri_det_rate = float(f.read().strip())

    sci_det_file = model_dir / 'sci_deterioration_rate.txt'
    sci_det_rate = 4.2
    if sci_det_file.exists():
        with open(sci_det_file, 'r') as f:
            sci_det_rate = float(f.read().strip())

    # Section 1: Surface, Traffic & Climate Data
    print("--- Section 1: Historical Surface, Traffic & Climate Data ---")
    shrp_id = input('SHRP Section ID (e.g., 0101): ').strip() or 'UNKNOWN'
    measurement_year = int(input('Historical Measurement Year (e.g., 2012 or 2018): '))
    mri = float(input('Measured IRI/MRI at that time (e.g., 0.85): '))
    aadtt = float(input('Daily Truck Count AADTT (e.g., 950): '))
    annual_truck_vol = float(input('Annual Truck Volume (e.g., 346750): '))
    annual_esal = float(input('Annual ESAL (e.g., 310000): '))
    cumulative_esal = float(input('Cumulative ESAL (e.g., 1500000): '))
    
    mean_temp = float(input('Mean Annual Temperature (°C) (e.g., 15.5): '))
    freeze_index = float(input('Annual Freeze Index (e.g., 10): '))
    freeze_thaw = float(input('Annual Freeze-Thaw Cycles (e.g., 45): '))

    features_iri = [
        'MRI', 'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND', 
        'ANNUAL_ESAL_TREND', 'CUMULATIVE_ESAL', 'YEAR',
        'MEAN_ANN_TEMP_AVG', 'FREEZE_INDEX_YR', 'FREEZE_THAW_YR'
    ]

    features_sci = [
        'SCI', 'BDI', 'DROP_LOAD', 'DROP_HEIGHT',
        'PAVEMENT_FAMILY_ENC', 'LANE_NO_ENC',
        'AADTT_ALL_TRUCKS_TREND', 'ANNUAL_TRUCK_VOLUME_TREND', 'ANNUAL_ESAL_TREND',
        'CUMULATIVE_ESAL', 'YEAR', 'YEARS_SINCE_LAST_REPAIR',
        'MEAN_ANN_TEMP_AVG', 'FREEZE_INDEX_YR', 'FREEZE_THAW_YR'
    ]

    historical_iri_score = score_iri(mri)

    # Section 2: Structural Data
    print("\n--- Section 2: Structural Data (Collected at Measurement Time) ---")
    has_fwd_input = input('Do you have FWD Deflection data for this measurement? (y/n): ').strip().lower()
    has_fwd = has_fwd_input == 'y'

    sci_val = None
    bdi_val = None
    historical_sci_score = None
    drop_load = 710.0
    drop_height = 4
    pav_enc = 0
    lane_enc = 0

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

            pav_enc = le_pav.transform([pav_family])[0]
            lane_enc = le_lane.transform([lane_no])[0]

            # AASHTO BELLS Temperature Normalization to standard 20°C
            # D_20 = D_t * 10^(-0.0079 * (20 - T_pavement))
            bells_factor = 10.0 ** (-0.0079 * (20.0 - mean_temp))
            d1_norm = deflections[0] * bells_factor
            d2_norm = deflections[1] * bells_factor

            # Mechanistic Indices: SCI (D1 - D2) and BDI (D2 - D3)
            sci_val = float(d1_norm - d2_norm)
            bdi_val = float(deflections[1] - deflections[2])
            historical_sci_score = score_sci(sci_val)

        except Exception as e:
            print(f"\nWarning: Could not process structural inputs ({e}). Falling back to surface-only.")
            has_fwd = False

    # 1. Historical Snapshot Calculation
    if has_fwd and historical_sci_score is not None:
        historical_rhi = float((historical_iri_score * 0.50) + (historical_sci_score * 0.50))
    else:
        historical_rhi = float(historical_iri_score)

    historical_condition = condition_label(historical_rhi)

    # 2. Synchronized AI Time-Series Fast-Forward to Present Day (2026)
    current_year = 2026
    current_mri = mri
    current_sci = sci_val if has_fwd else None
    current_bdi = bdi_val if has_fwd else None
    current_cum_esal = cumulative_esal
    years_since_repair = 0

    if measurement_year < current_year:
        print(f"\nAI Synchronously Forecasting Surface & Structure from {measurement_year} to {current_year}...")
        for yr in range(measurement_year, current_year):
            # Surface step prediction
            step_iri_input = pd.DataFrame([[
                current_mri, aadtt, annual_truck_vol, annual_esal, 
                current_cum_esal, yr, mean_temp, freeze_index, freeze_thaw
            ]], columns=features_iri)
            raw_next_mri = float(model_iri.predict(step_iri_input)[0])
            next_mri = raw_next_mri if raw_next_mri > current_mri else current_mri + iri_det_rate
            current_mri = next_mri

            # Structural step prediction (Annualized Delta + Virtual Maintenance Trigger)
            if has_fwd and current_sci is not None:
                step_sci_input = pd.DataFrame([[
                    current_sci, current_bdi, drop_load, drop_height,
                    pav_enc, lane_enc,
                    aadtt, annual_truck_vol, annual_esal,
                    current_cum_esal, yr, years_since_repair,
                    mean_temp, freeze_index, freeze_thaw
                ]], columns=features_sci)
                try:
                    predicted_delta = float(model_sci.predict(step_sci_input)[0])
                    annual_deg = max(1.5, min(predicted_delta, 8.0))
                except Exception:
                    annual_deg = sci_det_rate

                next_sci = current_sci + annual_deg

                # Virtual Maintenance Trigger
                if next_sci > 150.0:
                    next_sci = 40.0
                    years_since_repair = 0
                    if current_bdi is not None:
                        current_bdi = max(15.0, current_bdi * 0.5)
                else:
                    years_since_repair += 1

                if current_bdi is not None and next_sci != 40.0:
                    current_bdi = current_bdi * (next_sci / max(1.0, current_sci))
                current_sci = next_sci

            current_cum_esal += annual_esal

    # 3. Present Day Score (2026 Synchronized 50/50 Evaluation)
    present_iri_score = score_iri(current_mri)
    
    if has_fwd and current_sci is not None:
        present_sci_score = score_sci(current_sci)
        present_rhi = float((present_iri_score * 0.50) + (present_sci_score * 0.50))
        structural_policy = "Synchronized 50/50 Dual AI Forecast (Surface + Structural Decay Projected to 2026)"
    else:
        present_sci_score = None
        present_rhi = present_iri_score
        structural_policy = "Surface AI Forecast (FWD not provided)"

    present_condition = condition_label(present_rhi)

    # --- DISPLAY DUAL REPORT ---
    print(f"\n{'='*60}")
    print(f" ROAD HEALTH INDEX (RHI) SYNCHRONIZED FORECAST REPORT")
    print(f" Section: {shrp_id}")
    print(f"{'='*60}")
    print(f"1. HISTORICAL SNAPSHOT ({measurement_year}):")
    print(f"   - Measured Surface IRI : {mri:.3f} m/km (Score: {historical_iri_score:.1f}/100)")
    if has_fwd and sci_val is not None:
        print(f"   - Measured Fatigue SCI : {sci_val:.1f} um (Score: {historical_sci_score:.1f}/100)")
    else:
        print(f"   - Structural Health    : N/A")
    print(f"   - Historical RHI Score : {historical_rhi:.2f} / 100 ({historical_condition})")
    print(f"{'-'*60}")
    print(f"2. PRESENT DAY ESTIMATION ({current_year}):")
    print(f"   - AI Simulated Years   : {max(0, current_year - measurement_year)} years")
    print(f"   - Estimated 2026 IRI   : {current_mri:.3f} m/km (Score: {present_iri_score:.1f}/100, Change: {current_mri - mri:+.3f} m/km)")
    if has_fwd and current_sci is not None:
        print(f"   - Estimated 2026 SCI   : {current_sci:.1f} um (Score: {present_sci_score:.1f}/100, Change: {current_sci - sci_val:+.1f} um)")
    print(f"   - Present Day RHI      : {present_rhi:.2f} / 100 ({present_condition})")
    print(f"   - Structural Policy    : {structural_policy}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()