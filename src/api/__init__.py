"""
FastAPI Model Serving API
Serves ML predictions for cryptocurrency price forecasting.
"""

from src.api.main import app
from src.api.predict import PredictionService

__all__ = ['app', 'PredictionService']
