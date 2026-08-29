import os
import sys
import json
import urllib.request
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_online_offline():
    print("==================================================")
    print("VERIFICATION OF ONLINE VS OFFLINE INFERENCE MODES")
    print("==================================================\n")

    sample_patient = {
        "age": 55.5,
        "gender": 1,
        "height": 165.0,
        "weight": 70.0,
        "ap_hi": 140,
        "ap_lo": 90,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }

    # 1. ONLINE MODE TEST (FastAPI /predict endpoint)
    base_url = 'http://127.0.0.1:8000'
    data = json.dumps(sample_patient).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(f"{base_url}/predict", data=data, headers=headers)

    with urllib.request.urlopen(req) as response:
        online_res = json.loads(response.read().decode())
    
    print("ONLINE MODE Result (FastAPI):")
    print("  Predicted Class:", online_res['predicted_class'])
    print("  Probability    :", online_res['probability'], f"({online_res['probability']*100:.2f}%)")

    # 2. OFFLINE MODE TEST (Local Standalone Weights & Scaler)
    with open(os.path.join(project_root, 'models', 'scaler_params.json'), 'r') as f:
        scaler_params = json.load(f)
    
    with open(os.path.join(project_root, 'models', 'global_model_weights.json'), 'r') as f:
        weights = json.load(f)

    # Local Preprocessing
    feature_cols = scaler_params['feature_columns']
    cont_cols = scaler_params['continuous_columns']
    means = scaler_params['mean']
    scales = scaler_params['scale']

    x_input = []
    for fname in feature_cols:
        val = sample_patient[fname]
        if fname in cont_cols:
            idx = cont_cols.index(fname)
            val = (val - means[idx]) / scales[idx]
        x_input.append(val)
    
    x_input = np.array(x_input, dtype=np.float32)

    # Local Forward Pass
    fc1_w = np.array(weights['fc1_w'], dtype=np.float32)
    fc1_b = np.array(weights['fc1_b'], dtype=np.float32)
    fc2_w = np.array(weights['fc2_w'], dtype=np.float32)
    fc2_b = np.array(weights['fc2_b'], dtype=np.float32)
    fc3_w = np.array(weights['fc3_w'], dtype=np.float32)
    fc3_b = np.array(weights['fc3_b'], dtype=np.float32)

    h1 = np.maximum(0.0, np.dot(fc1_w, x_input) + fc1_b)
    h2 = np.maximum(0.0, np.dot(fc2_w, h1) + fc2_b)
    logit = float(np.dot(fc3_w, h2) + fc3_b)

    offline_prob = float(1.0 / (1.0 + np.exp(-logit)))
    offline_class = 1 if offline_prob >= 0.5 else 0
    offline_prob_rounded = round(offline_prob, 4)

    print("\nOFFLINE MODE Result (Local Engine):")
    print("  Predicted Class:", offline_class)
    print("  Probability    :", offline_prob_rounded, f"({offline_prob_rounded*100:.2f}%)")

    # Equivalence Verification
    diff = abs(online_res['probability'] - offline_prob_rounded)
    print(f"\nAbsolute Difference: {diff:.6f}")

    if online_res['predicted_class'] == offline_class and diff < 1e-4:
        print("\nMODES EQUIVALENCE VERIFICATION: PASS (100% Identical Output!)")
    else:
        print("\nMODES EQUIVALENCE VERIFICATION: FAIL")

if __name__ == '__main__':
    test_online_offline()
