import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.monitoring.metrics import collector
from src.serving.model_loader import ModelArtifacts, load_production_artifacts
from src.serving.schemas import CustomerInput, PredictionResponse

INDEX_HTML_PATH = Path(__file__).parent / "static" / "index.html"

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


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Telco Churn Prediction API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/health")
def health_check():
    if "current" not in artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return {
        "status": "healthy",
        "model_version": artifacts["current"].version,
    }


@app.get("/metrics")
def get_metrics():
    version = artifacts["current"].version if "current" in artifacts else "unknown"
    return collector.get_summary(model_version=version)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if "current" not in artifacts:
        collector.record_request(latency_ms=0.0, is_error=True)
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    start_time = time.perf_counter()
    art = artifacts["current"]

    try:
        # 1. Convert validated Pydantic model into a single-row DataFrame
        input_dict = customer.model_dump()
        input_df = pd.DataFrame([input_dict])

        # 2. Transform using the paired training preprocessor
        features = art.preprocessor.transform(input_df)

        # 3. Predict probability
        churn_proba = float(art.model.predict_proba(features)[0, 1])
        churn_pred = int(churn_proba >= art.threshold)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        collector.record_request(latency_ms=latency_ms, prediction=churn_pred)

        return PredictionResponse(
            churn_prediction=churn_pred,
            churn_probability=round(churn_proba, 4),
            threshold=art.threshold,
            model_version=art.version,
        )
    except Exception as exc:
        collector.record_request(latency_ms=0.0, is_error=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
