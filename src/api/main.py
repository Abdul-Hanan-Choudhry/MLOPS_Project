"""
FastAPI Application for Crypto Price Prediction Model Serving.

This module provides a REST API for:
- Model predictions
- Health checks
- Model information
- Prometheus metrics for monitoring
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.predict import PredictionService
from src.api.metrics import (
    get_metrics,
    get_content_type,
    MetricsMiddleware,
    DriftDetector,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ACTIVE_REQUESTS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class AlertWebhookPayload(BaseModel):
    """Payload for Grafana alert webhooks."""
    alerts: List[Dict[str, Any]] = []
    status: str = ""
    commonLabels: Dict[str, str] = {}


# Store alerts for logging
alert_log: List[Dict] = []


@app.on_event("startup")
async def startup_event():
    """Initialize the prediction service on startup."""
    global prediction_service
    try:
        prediction_service = PredictionService()
        logger.info(f"✅ Model loaded: {prediction_service.model_name}")
        
        # Set model info in metrics
        MetricsMiddleware.set_model_info(
            name=prediction_service.model_name,
            model_type=type(prediction_service.model).__name__,
            version="1.0.0"
        )
    except Exception as e:
        logger.warning(f"⚠️ Failed to load model: {e}")
        prediction_service = None


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to track request metrics."""
    start_time = time.time()
    
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        status = "success" if response.status_code < 400 else "error"
    except Exception as e:
        status = "error"
        raise
    finally:
        duration = time.time() - start_time
        endpoint = request.url.path
        method = request.method
        
        ACTIVE_REQUESTS.dec()
        MetricsMiddleware.track_request(method, endpoint, status, duration)
    
    return response


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
    Also tracks metrics and checks for data drift.
    """
    if not prediction_service:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        # Check for data drift
        ood_status = DriftDetector.check_features(request.features)
        
        # Make prediction
        prediction = prediction_service.predict(request.features)
        
        # Track prediction metrics
        MetricsMiddleware.track_prediction(
            model_name=prediction_service.model_name,
            prediction_value=prediction
        )
        
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


# ============================================================================
# PROMETHEUS METRICS ENDPOINT
# ============================================================================

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes all collected metrics in Prometheus format.
    """
    return Response(
        content=get_metrics(),
        media_type=get_content_type()
    )


# ============================================================================
# ALERTING ENDPOINTS
# ============================================================================

@app.post("/alerts/webhook")
async def receive_alert(payload: AlertWebhookPayload):
    """
    Webhook endpoint to receive Grafana alerts.
    
    Logs alerts to a file and stores them in memory.
    """
    timestamp = datetime.now().isoformat()
    
    alert_entry = {
        "timestamp": timestamp,
        "status": payload.status,
        "labels": payload.commonLabels,
        "alerts": payload.alerts
    }
    
    # Store in memory
    alert_log.append(alert_entry)
    
    # Keep only last 100 alerts
    if len(alert_log) > 100:
        alert_log.pop(0)
    
    # Log to file
    log_dir = os.environ.get("LOG_DIR", "/app/logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "alerts.log")
    try:
        with open(log_file, "a") as f:
            f.write(f"{timestamp} | {payload.status} | {payload.commonLabels}\n")
            for alert in payload.alerts:
                f.write(f"  - {alert.get('labels', {}).get('alertname', 'unknown')}: "
                       f"{alert.get('annotations', {}).get('summary', 'No summary')}\n")
    except Exception as e:
        logger.warning(f"Failed to write alert log: {e}")
    
    logger.info(f"🚨 Alert received: {payload.status} - {payload.commonLabels}")
    
    return {"status": "received", "timestamp": timestamp}


@app.get("/alerts/history")
async def get_alert_history():
    """
    Get recent alert history.
    
    Returns the last 100 alerts received.
    """
    return {
        "alerts": alert_log,
        "count": len(alert_log),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# DRIFT MONITORING ENDPOINTS
# ============================================================================

@app.get("/drift/status")
async def drift_status():
    """
    Get current data drift status.
    
    Returns OOD ratios for all monitored features.
    """
    from src.api.metrics import DriftDetector
    
    return {
        "feature_stats": DriftDetector._feature_stats,
        "ood_counts": DriftDetector._ood_counts,
        "total_counts": DriftDetector._total_counts,
        "timestamp": datetime.now().isoformat()
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
