import os
import sys
import pickle
import pandas as pd
import torch

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.model import DiagnosticModel
from preprocessing.preprocess import transform_new_patient, FEATURE_COLUMNS
from backend.schemas import PatientPredictRequest, PredictionResponse, ModelInfoResponse


class ModelService:
    """
    Service class handling global DiagnosticModel loading and single-patient inference.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.load_model_and_scaler()

    def load_model_and_scaler(self):
        """
        Loads the final saved global model state_dict and reference StandardScaler.
        """
        models_dir = os.path.join(project_root, 'models')
        model_path = os.path.join(models_dir, 'global_model.pt')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Saved global model weights not found at: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Saved scaler not found at: {scaler_path}")

        # 1. Instantiate PyTorch DiagnosticModel (CPU mode)
        self.model = DiagnosticModel(input_size=11, hidden_size1=64, hidden_size2=32)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 2. Load fitted scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.is_loaded = True
        print("ModelService: Global DiagnosticModel and StandardScaler loaded successfully.")

    def predict(self, request: PatientPredictRequest) -> PredictionResponse:
        """
        Executes real-time CPU inference for a single patient record.
        """
        if not self.is_loaded:
            raise RuntimeError("ModelService is not loaded properly.")

        # Convert Pydantic request to pandas DataFrame with exact canonical feature order
        patient_dict = request.model_dump()
        df = pd.DataFrame([patient_dict])[FEATURE_COLUMNS]

        # Preprocess features using the existing fitted scaler
        X_tensor = transform_new_patient(df, self.scaler)

        # Run CPU inference under no_grad context
        with torch.no_grad():
            logit = self.model(X_tensor)
            prob = torch.sigmoid(logit).item()
            pred_class = 1 if prob >= 0.5 else 0

        return PredictionResponse(
            predicted_class=pred_class,
            probability=round(prob, 4),
            model_type="Federated FNN"
        )

    def get_info(self) -> ModelInfoResponse:
        """
        Returns technical metadata about the deployed federated model.
        """
        return ModelInfoResponse(
            model_architecture="DiagnosticModel (11 -> 64 -> 32 -> 1)",
            federated_rounds=5,
            participating_clinics=["Clinic A", "Clinic B", "Clinic C"],
            model_status="Loaded & Operational"
        )


# Global singleton instance
model_service = ModelService()
