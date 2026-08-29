import os
import sys
import json
import pickle
import urllib.request
import urllib.error
import torch
import pandas as pd

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.model import DiagnosticModel
from preprocessing.preprocess import transform_new_patient, FEATURE_COLUMNS

def run_connection_audit():
    print("==================================================")
    print("END-TO-END CONNECTION AUDIT & VERIFICATION")
    print("==================================================\n")

    base_url = 'http://127.0.0.1:8000'
    audit_results = {}

    # 1. Test GET /health
    try:
        req = urllib.request.urlopen(f"{base_url}/health")
        health_data = json.loads(req.read().decode())
        if health_data.get('status') == 'ok':
            audit_results['/health'] = 'PASS'
            print("1. GET /health: PASS ->", health_data)
        else:
            audit_results['/health'] = 'FAIL'
    except Exception as e:
        audit_results['/health'] = f'FAIL: {e}'

    # 2. Test GET /model-info
    try:
        req = urllib.request.urlopen(f"{base_url}/model-info")
        info_data = json.loads(req.read().decode())
        if info_data.get('model_status') == 'Loaded & Operational' and info_data.get('federated_rounds') == 5:
            audit_results['/model-info'] = 'PASS'
            print("2. GET /model-info: PASS ->", info_data)
        else:
            audit_results['/model-info'] = 'FAIL'
    except Exception as e:
        audit_results['/model-info'] = f'FAIL: {e}'

    # 3. Test POST /predict via REST API
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

    try:
        data = json.dumps(sample_patient).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        post_req = urllib.request.Request(f"{base_url}/predict", data=data, headers=headers)
        
        with urllib.request.urlopen(post_req) as response:
            api_pred = json.loads(response.read().decode())
            audit_results['/predict'] = 'PASS'
            print("3. POST /predict REST API: PASS ->", api_pred)
    except Exception as e:
        audit_results['/predict'] = f'FAIL: {e}'

    # 4. Direct Local PyTorch & Preprocessor Equivalence Verification
    print("\n--- Direct PyTorch vs FastAPI Equivalence Verification ---")
    try:
        model_path = os.path.join(project_root, 'models', 'global_model.pt')
        scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')

        # Load direct model & scaler
        direct_model = DiagnosticModel(input_size=11, hidden_size1=64, hidden_size2=32)
        direct_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        direct_model.eval()

        with open(scaler_path, 'rb') as f:
            direct_scaler = pickle.load(f)

        df = pd.DataFrame([sample_patient])[FEATURE_COLUMNS]
        X_tensor = transform_new_patient(df, direct_scaler)

        with torch.no_grad():
            logit = direct_model(X_tensor)
            direct_prob = torch.sigmoid(logit).item()
            direct_class = 1 if direct_prob >= 0.5 else 0

        direct_prob_rounded = round(direct_prob, 4)

        print(f"Direct PyTorch Inference Output : predicted_class={direct_class}, probability={direct_prob_rounded}")
        print(f"FastAPI /predict Endpoint Output: predicted_class={api_pred['predicted_class']}, probability={api_pred['probability']}")

        if direct_class == api_pred['predicted_class'] and direct_prob_rounded == api_pred['probability']:
            audit_results['End-to-End Prediction Match'] = 'PASS'
            print("Direct vs API Equivalence Check: PASS (100% Exact Match)")
        else:
            audit_results['End-to-End Prediction Match'] = 'FAIL'
    except Exception as e:
        audit_results['End-to-End Prediction Match'] = f'FAIL: {e}'

    # 5. Error Handling Audit (Invalid payload)
    print("\n--- Error Handling Audit ---")
    try:
        invalid_patient = {"age": 55.5} # missing required fields
        data = json.dumps(invalid_patient).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        bad_req = urllib.request.Request(f"{base_url}/predict", data=data, headers=headers)
        urllib.request.urlopen(bad_req)
        audit_results['Error Handling'] = 'FAIL'
    except urllib.error.HTTPError as e:
        if e.code == 422:
            audit_results['Error Handling'] = 'PASS'
            print("Error Handling Check: PASS (422 Unprocessable Entity gracefully returned)")
        else:
            audit_results['Error Handling'] = f'FAIL: Code {e.code}'

    print("\n==================================================")
    print("AUDIT SUMMARY REPORT")
    print("==================================================")
    for k, v in audit_results.items():
        print(f"{k:<35}: {v}")

if __name__ == '__main__':
    run_connection_audit()
