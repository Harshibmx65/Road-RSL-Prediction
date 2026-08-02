import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

def main():
    print("Loading data for Model 2 (FWD Unsupervised Clustering)...")
    
    # --- ROBUST PATH RESOLUTION ---
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    model_dir = project_root / 'models'
    model_dir.mkdir(exist_ok=True)
    # ------------------------------

    fwd = pd.read_excel(data_dir / "MON_DEFL_DROP_DATA.xlsx")
    exp = pd.read_excel(data_dir / "EXPERIMENT_SECTION.xlsx")

    # Clean & Merge
    fwd['SHRP_ID'] = fwd['SHRP_ID'].astype(str).str.zfill(4)
    exp_clean = exp[['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO', 'PAVEMENT_FAMILY']].drop_duplicates()

    columns_to_drop = ['PEAK_DEFL_8', 'PEAK_DEFL_9', 'NON_DECREASING_DEFL', 
                       'NON_DECREASING_DEFL_EXP', 'STATE_CODE_EXP', 
                       'LANE_NO_EXP', 'DROP_HEIGHT_EXP', 'DEFL_UNIT_ID']
    fwd = fwd.drop(columns=columns_to_drop, errors='ignore')

    df2 = pd.merge(fwd, exp_clean, on=['SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'], how='inner')
    df2 = df2.dropna()

    # Encode categorical features
    le_pav = LabelEncoder()
    df2['PAVEMENT_FAMILY_ENC'] = le_pav.fit_transform(df2['PAVEMENT_FAMILY'])

    le_lane = LabelEncoder()
    df2['LANE_NO_ENC'] = le_lane.fit_transform(df2['LANE_NO'])

    features2 = ['PEAK_DEFL_1', 'PEAK_DEFL_2', 'PEAK_DEFL_3', 'PEAK_DEFL_4', 
                 'PEAK_DEFL_5', 'PEAK_DEFL_6', 'PEAK_DEFL_7', 'DROP_LOAD', 
                 'DROP_HEIGHT', 'PAVEMENT_FAMILY_ENC', 'LANE_NO_ENC']

    X2 = df2[features2]

    # Standardize data for K-Means
    scaler = StandardScaler()
    X2_scaled = scaler.fit_transform(X2)

    print("Training K-Means Model...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df2['CLUSTER'] = kmeans.fit_predict(X2_scaled)

    # Dynamic Health Mapping
    df2['AVG_DEFL'] = df2[['PEAK_DEFL_1','PEAK_DEFL_2','PEAK_DEFL_3',
                           'PEAK_DEFL_4','PEAK_DEFL_5','PEAK_DEFL_6',
                           'PEAK_DEFL_7']].mean(axis=1)
    
    cluster_means = df2.groupby('CLUSTER')['AVG_DEFL'].mean().sort_values()
    health_mapping = {
        cluster_means.index[0]: 'Good',
        cluster_means.index[1]: 'Fair',
        cluster_means.index[2]: 'Poor'
    }

    # Save the artifacts
    joblib.dump(kmeans, model_dir / 'fwd_kmeans_model.pkl')
    joblib.dump(scaler, model_dir / 'fwd_scaler.pkl')
    joblib.dump(le_pav, model_dir / 'fwd_le_pav.pkl')
    joblib.dump(le_lane, model_dir / 'fwd_le_lane.pkl')
    joblib.dump(health_mapping, model_dir / 'fwd_health_mapping.pkl')
    
    print("Model 2 and preprocessors saved to models directory!")

if __name__ == "__main__":
    main()