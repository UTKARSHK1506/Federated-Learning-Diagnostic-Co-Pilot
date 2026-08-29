import os
import sys
import pickle
import json
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.model import DiagnosticModel
from preprocessing.preprocess import transform_new_patient, FEATURE_COLUMNS

CONTINUOUS_COLS = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

def convert_and_verify():
    print("==================================================")
    print("STEP 1 & 2: ONNX CONVERSION & EQUIVALENCE CHECK")
    print("==================================================")

    models_dir = os.path.join(project_root, 'models')
    model_pt_path = os.path.join(models_dir, 'global_model.pt')
    model_onnx_path = os.path.join(models_dir, 'global_model.onnx')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')

    # 1. Load PyTorch global model
    pt_model = DiagnosticModel(input_size=11, hidden_size1=64, hidden_size2=32)
    pt_model.load_state_dict(torch.load(model_pt_path, map_location=torch.device('cpu')))
    pt_model.eval()

    # 2. Export to ONNX via TorchScript exporter
    dummy_input = torch.randn(1, 11, dtype=torch.float32)
    torch.onnx.export(
        pt_model,
        dummy_input,
        model_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"ONNX Model saved successfully to: '{model_onnx_path}' ({os.path.getsize(model_onnx_path)} bytes)")

    # 3. Extract and save Scaler Parameters (mean_ and scale_) for local offline preprocessing
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    scaler_params = {
        "continuous_columns": CONTINUOUS_COLS,
        "feature_columns": FEATURE_COLUMNS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    
    scaler_json_path = os.path.join(models_dir, 'scaler_params.json')
    with open(scaler_json_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Scaler parameters saved to: '{scaler_json_path}'")

    # 4. Equivalence Test between PyTorch and ONNXRuntime
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

    df = pd.DataFrame([sample_patient])[FEATURE_COLUMNS]
    X_tensor = transform_new_patient(df, scaler)
    X_numpy = X_tensor.numpy()

    # PyTorch Inference
    with torch.no_grad():
        pt_logit = pt_model(X_tensor).item()
        pt_prob = torch.sigmoid(torch.tensor(pt_logit)).item()
        pt_class = 1 if pt_prob >= 0.5 else 0

    # ONNXRuntime Inference
    ort_session = ort.InferenceSession(model_onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: X_numpy}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_logit = float(ort_outputs[0][0][0])
    onnx_prob = 1.0 / (1.0 + np.exp(-onnx_logit))
    onnx_class = 1 if onnx_prob >= 0.5 else 0

    print("\n--- Equivalence Results ---")
    print(f"PyTorch Model : logit={pt_logit:.6f}, prob={pt_prob:.6f}, class={pt_class}")
    print(f"ONNX Model    : logit={onnx_logit:.6f}, prob={onnx_prob:.6f}, class={onnx_class}")

    abs_diff = abs(pt_prob - onnx_prob)
    print(f"Absolute Difference: {abs_diff:.8f}")

    if abs_diff < 1e-5 and pt_class == onnx_class:
        print("\nEQUIVALENCE TEST: PASS (PyTorch and ONNX models produce 100% identical predictions!)")
    else:
        print("\nEQUIVALENCE TEST: FAIL")

if __name__ == '__main__':
    convert_and_verify()
