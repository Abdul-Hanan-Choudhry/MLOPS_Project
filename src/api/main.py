"""
FastAPI Application for Crypto Price Prediction Model Serving.

This module provides a REST API for:
- Model predictions
- Health checks
- Model information
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.predict import PredictionService

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API",
    description="REST API for Bitcoin price prediction using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global prediction service
prediction_service: Optional[PredictionService] = None


# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    model_name: Optional[str] = None
    version: str = "1.0.0"


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names to values",
        example={
            "price": 95000.0,
            "volume": 50000000000.0,
            "market_cap": 1900000000000.0,
            "price_lag_1": 94500.0,
            "price_lag_3": 94000.0,
            "price_rolling_mean_6": 94200.0,
            "price_rolling_std_6": 500.0
        }
    )


class BatchPredictionRequest(BaseModel):
    instances: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries for batch prediction"
    )


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    timestamp: str
    confidence: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_name: str
    timestamp: str
    count: int


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    features: List[str]
    metrics: Dict[str, float]
    trained_at: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize the prediction service on startup."""
    global prediction_service
    try:
        prediction_service = PredictionService()
        print(f"✅ Model loaded: {prediction_service.model_name}")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
        prediction_service = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Crypto Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and model loading state.
    """
    return HealthResponse(
        status="healthy" if prediction_service else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=prediction_service is not None,
        model_name=prediction_service.model_name if prediction_service else None,
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.
    
    Returns model metadata including features, metrics, and version.
    """
    if not prediction_service:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    info = prediction_service.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction.
    
    Accepts feature values and returns the predicted price.
    """
    if not prediction_service:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        prediction = prediction_service.predict(request.features)
        return PredictionResponse(
            prediction=prediction,
            model_name=prediction_service.model_name,
            timestamp=datetime.now().isoformat()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Accepts multiple instances and returns predictions for all.
    """
    if not prediction_service:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        predictions = prediction_service.predict_batch(request.instances)
        return BatchPredictionResponse(
            predictions=predictions,
            model_name=prediction_service.model_name,
            timestamp=datetime.now().isoformat(),
            count=len(predictions)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """
    Reload the model from disk.
    
    Useful after model updates without restarting the service.
    """
    global prediction_service
    
    try:
        prediction_service = PredictionService()
        return {
            "status": "success",
            "message": f"Model reloaded: {prediction_service.model_name}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
