from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.serving.model_loader import ModelArtifacts, load_production_artifacts
from src.serving.schemas import CustomerInput, PredictionResponse

# Global container for loaded artifacts
artifacts: dict[str, ModelArtifacts] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and preprocessor ONCE at application startup
    try:
        artifacts["current"] = load_production_artifacts()
        print(f"Loaded CustomerChurnModel Version {artifacts['current'].version} (@production)")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load production model: {e}")
    yield
    artifacts.clear()


app = FastAPI(title="Telco Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    if "current" not in artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return {
        "status": "healthy",
        "model_version": artifacts["current"].version,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if "current" not in artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    art = artifacts["current"]

    # 1. Convert validated Pydantic model into a single-row DataFrame
    input_dict = customer.model_dump()
    input_df = pd.DataFrame([input_dict])

    # 2. Transform using the paired training preprocessor
    features = art.preprocessor.transform(input_df)

    # 3. Predict probability
    churn_proba = float(art.model.predict_proba(features)[0, 1])
    churn_pred = int(churn_proba >= art.threshold)

    return PredictionResponse(
        churn_prediction=churn_pred,
        churn_probability=round(churn_proba, 4),
        threshold=art.threshold,
        model_version=art.version,
    )
