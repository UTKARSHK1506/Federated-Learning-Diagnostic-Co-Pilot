from pydantic import BaseModel, Field
from typing import List


class PatientPredictRequest(BaseModel):
    """
    Pydantic schema for patient clinical feature input payload.
    """
    age: float = Field(..., description="Patient age in years", example=55.5)
    gender: int = Field(..., description="Gender (1 = female, 2 = male)", example=1)
    height: float = Field(..., description="Height in centimeters", example=165.0)
    weight: float = Field(..., description="Weight in kilograms", example=70.0)
    ap_hi: int = Field(..., description="Systolic blood pressure (mmHg)", example=130)
    ap_lo: int = Field(..., description="Diastolic blood pressure (mmHg)", example=85)
    cholesterol: int = Field(..., description="Cholesterol level (1 = normal, 2 = above normal, 3 = well above normal)", example=1)
    gluc: int = Field(..., description="Glucose level (1 = normal, 2 = above normal, 3 = well above normal)", example=1)
    smoke: int = Field(..., description="Smoking status (0 = non-smoker, 1 = smoker)", example=0)
    alco: int = Field(..., description="Alcohol intake (0 = non-drinker, 1 = drinker)", example=0)
    active: int = Field(..., description="Physical activity status (0 = non-active, 1 = active)", example=1)


class PredictionResponse(BaseModel):
    """
    Pydantic schema for inference output response.
    """
    predicted_class: int = Field(..., description="Binary diagnostic risk prediction (0 = Low Risk, 1 = High Risk)")
    probability: float = Field(..., description="Predicted probability of cardiovascular risk (0.0 to 1.0)")
    model_type: str = Field(..., description="Type of model used for inference")


class ModelInfoResponse(BaseModel):
    """
    Pydantic schema for model information response.
    """
    model_architecture: str
    federated_rounds: int
    participating_clinics: List[str]
    model_status: str


class HealthResponse(BaseModel):
    """
    Pydantic schema for health check status.
    """
    status: str
