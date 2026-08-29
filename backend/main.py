import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.schemas import PatientPredictRequest, PredictionResponse, ModelInfoResponse, HealthResponse
from backend.model_service import model_service

app = FastAPI(
    title="CardioSense API",
    description="REST API service for single-patient cardiovascular disease risk prediction using the global federated model.",
    version="1.0.0"
)

# Enable CORS for Android application and external client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint returning system status.
    """
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def get_model_info():
    """
    Returns information about the deployed federated global model.
    """
    return model_service.get_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_cardio_risk(patient: PatientPredictRequest):
    """
    Predicts cardiovascular disease risk for a single patient record.
    """
    return model_service.predict(patient)


# Serve Mobile App UI
app_dir = os.path.join(project_root, 'app')
if os.path.exists(app_dir):
    app.mount("/static", StaticFiles(directory=app_dir), name="static")

    @app.get("/", tags=["UI"])
    def read_root():
        return FileResponse(os.path.join(app_dir, "index.html"))

    @app.get("/style.css", tags=["UI"])
    def get_css():
        return FileResponse(os.path.join(app_dir, "style.css"))

    @app.get("/script.js", tags=["UI"])
    def get_js():
        return FileResponse(os.path.join(app_dir, "script.js"))

    @app.get("/global_model_weights.json", tags=["UI"])
    def get_weights():
        return FileResponse(os.path.join(app_dir, "global_model_weights.json"))

    @app.get("/scaler_params.json", tags=["UI"])
    def get_scaler():
        return FileResponse(os.path.join(app_dir, "scaler_params.json"))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
