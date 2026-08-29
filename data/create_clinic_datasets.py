import os
import pandas as pd
import numpy as np

def create_clinic_datasets():
    # Set fixed random seed for 100% reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    # 1. Define paths and read preprocessed dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'cardio_cleaned.csv')
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found at: {input_path}")
        
    print(f"Reading preprocessed dataset from: {input_path}")
    # Detect delimiter (comma or semicolon)
    df = pd.read_csv(input_path, sep=None, engine='python')
    
    total_rows, total_cols = df.shape
    print(f"Dataset loaded: {total_rows} rows, {total_cols} columns.")
    print(f"Columns: {df.columns.tolist()}")
    
    # 2. Non-IID Partition Strategy based on Age and Blood Pressure
    # Calculate standardized age and ap_hi for ranking score
    age_norm = (df['age'] - df['age'].mean()) / df['age'].std()
    
    # Clip ap_hi to robust percentiles ONLY for ranking score calculation (original CSV values remain untouched)
    ap_hi_clipped = df['ap_hi'].clip(lower=df['ap_hi'].quantile(0.01), upper=df['ap_hi'].quantile(0.99))
    ap_hi_norm = (ap_hi_clipped - ap_hi_clipped.mean()) / ap_hi_clipped.std()
    
    # Composite score: higher score indicates older age & higher blood pressure
    composite_score = age_norm + ap_hi_norm + np.random.normal(0, 0.75, size=total_rows)
    
    # Sort dataset indices by composite score
    sorted_indices = np.argsort(composite_score)
    
    # Divide into lower-score pool and higher-score pool
    midpoint = total_rows // 2
    lower_pool = sorted_indices[:midpoint]
    higher_pool = sorted_indices[midpoint:]
    
    # Target sizes: ~1/3 each
    size_a = total_rows // 3 + (total_rows % 3 > 0)
    size_b = total_rows // 3 + (total_rows % 3 > 1)
    size_c = total_rows - size_a - size_b
    
    # Clinic A (Older + Higher BP): 78% from higher_pool, 22% from lower_pool
    count_a_high = int(size_a * 0.78)
    count_a_low = size_a - count_a_high
    
    a_high_selected = np.random.choice(higher_pool, size=count_a_high, replace=False)
    a_low_selected = np.random.choice(lower_pool, size=count_a_low, replace=False)
    clinic_a_indices = np.concatenate([a_high_selected, a_low_selected])
    
    # Remaining pools
    remaining_higher = np.setdiff1d(higher_pool, a_high_selected)
    remaining_lower = np.setdiff1d(lower_pool, a_low_selected)
    
    # Clinic B (Younger + Lower BP): 78% from remaining_lower, 22% from remaining_higher
    count_b_low = int(size_b * 0.78)
    count_b_high = size_b - count_b_low
    
    b_low_selected = np.random.choice(remaining_lower, size=count_b_low, replace=False)
    b_high_selected = np.random.choice(remaining_higher, size=count_b_high, replace=False)
    clinic_b_indices = np.concatenate([b_low_selected, b_high_selected])
    
    # Clinic C (Mixed population): Remaining unassigned rows from both pools
    remaining_higher_final = np.setdiff1d(remaining_higher, b_high_selected)
    remaining_lower_final = np.setdiff1d(remaining_lower, b_low_selected)
    clinic_c_indices = np.concatenate([remaining_higher_final, remaining_lower_final])
    
    # Shuffle each clinic's indices to avoid ordering bias
    np.random.shuffle(clinic_a_indices)
    np.random.shuffle(clinic_b_indices)
    np.random.shuffle(clinic_c_indices)
    
    # Extract sub-dataframes
    df_a = df.iloc[clinic_a_indices].copy()
    df_b = df.iloc[clinic_b_indices].copy()
    df_c = df.iloc[clinic_c_indices].copy()
    
    # 3. Save datasets using standard comma delimiter
    path_a = os.path.join(base_dir, 'clinic_A.csv')
    path_b = os.path.join(base_dir, 'clinic_B.csv')
    path_c = os.path.join(base_dir, 'clinic_C.csv')
    
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)
    df_c.to_csv(path_c, index=False)
    
    print("\nSaved simulated clinic datasets successfully:")
    print(f"  - Clinic A: {path_a} ({len(df_a)} rows)")
    print(f"  - Clinic B: {path_b} ({len(df_b)} rows)")
    print(f"  - Clinic C: {path_c} ({len(df_c)} rows)")
    
    # 4. Verification and Non-IID Distribution Report
    print("\n" + "="*70)
    print("VERIFICATION AND NON-IID CLINIC DISTRIBUTION REPORT")
    print("="*70)
    
    # Check duplicate patient IDs across clinics
    ids_a = set(df_a['id'])
    ids_b = set(df_b['id'])
    ids_c = set(df_c['id'])
    
    overlap_ab = len(ids_a.intersection(ids_b))
    overlap_bc = len(ids_b.intersection(ids_c))
    overlap_ca = len(ids_c.intersection(ids_a))
    total_unique_ids = len(ids_a | ids_b | ids_c)
    
    print(f"\n--- 1. OVERLAP & DUPLICATION CHECK ---")
    print(f"Total Unique Patient IDs across all 3 clinics: {total_unique_ids} / {total_rows}")
    print(f"Overlap Clinic A & Clinic B: {overlap_ab} rows")
    print(f"Overlap Clinic B & Clinic C: {overlap_bc} rows")
    print(f"Overlap Clinic C & Clinic A: {overlap_ca} rows")
    print(f"Duplication Status: {'PASSED (Zero duplicates across clinics)' if (overlap_ab + overlap_bc + overlap_ca == 0 and total_unique_ids == total_rows) else 'FAILED'}")
    
    print(f"\n--- 2. CLINIC DATASET SUMMARY TABLE ---")
    clinics = [('Clinic A (Older / High BP)', df_a), 
               ('Clinic B (Younger / Low BP)', df_b), 
               ('Clinic C (Mixed Population)', df_c)]
    
    for name, c_df in clinics:
        # Determine if age is in years or days
        age_vals = c_df['age']
        age_years = age_vals if age_vals.mean() < 100 else age_vals / 365.25
        
        cardio_pos = (c_df['cardio'] == 1).sum()
        cardio_neg = (c_df['cardio'] == 0).sum()
        cardio_pct = (cardio_pos / len(c_df)) * 100
        
        bp_hi_filtered = c_df[(c_df['ap_hi'] >= 70) & (c_df['ap_hi'] <= 240)]['ap_hi']
        bp_lo_filtered = c_df[(c_df['ap_lo'] >= 40) & (c_df['ap_lo'] <= 140)]['ap_lo']
        
        print(f"\n>>> {name} <<<")
        print(f"  Rows: {len(c_df)} | Columns: {c_df.shape[1]}")
        print(f"  Target Distribution ('cardio'):")
        print(f"    - Negative (0): {cardio_neg} ({100 - cardio_pct:.2f}%)")
        print(f"    - Positive (1): {cardio_pos} ({cardio_pct:.2f}%)")
        print(f"  Age Statistics (years):")
        print(f"    - Mean: {age_years.mean():.2f} yrs | Std: {age_years.std():.2f}")
        print(f"    - Median: {age_years.median():.2f} yrs | Min: {age_years.min():.2f} | Max: {age_years.max():.2f}")
        print(f"  Systolic Blood Pressure ('ap_hi'):")
        print(f"    - Mean: {bp_hi_filtered.mean():.2f} mmHg | Median: {bp_hi_filtered.median():.2f} mmHg")
        print(f"  Diastolic Blood Pressure ('ap_lo'):")
        print(f"    - Mean: {bp_lo_filtered.mean():.2f} mmHg | Median: {bp_lo_filtered.median():.2f} mmHg")

if __name__ == '__main__':
    create_clinic_datasets()
