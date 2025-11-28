"""
Prediction Service Module.

Handles model loading and inference for the API.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import joblib


class PredictionService:
    """
    Service class for making predictions with the trained model.
    
    Handles:
    - Model loading from local files or MLflow
    - Feature preprocessing
    - Inference
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize the prediction service.
        
        Args:
            model_path: Path to the model file. If None, uses default location.
            scaler_path: Path to the scaler file. If None, uses default location.
        """
        self.model = None
        self.scaler = None
        self.metadata = {}
        self.model_name = "unknown"
        self.feature_names = []
        
        # Default paths
        models_dir = os.environ.get("MODELS_DIR", "/app/models")
        if not os.path.exists(models_dir):
            models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        
        self.model_path = model_path or self._find_model(models_dir)
        self.scaler_path = scaler_path or os.path.join(models_dir, "scaler.joblib")
        self.metadata_path = os.path.join(models_dir, "best_model_metadata.json")
        
        # Fallback to .pkl extensions
        if not os.path.exists(self.scaler_path):
            self.scaler_path = os.path.join(models_dir, "scaler.pkl")
        
        self._load_model()
    
    def _find_model(self, models_dir: str) -> str:
        """Find the best model file in the models directory."""
        models_dir = Path(models_dir)
        
        # Look for best_model files
        for pattern in ["best_model*.joblib", "best_model*.pkl", "*.joblib", "*.pkl"]:
            files = list(models_dir.glob(pattern))
            if files:
                # Return the most recently modified file
                return str(max(files, key=lambda x: x.stat().st_mtime))
        
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    def _load_model(self):
        """Load the model, scaler, and metadata."""
        print(f"Loading model from: {self.model_path}")
        
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        
        # Extract model name from filename
        model_filename = os.path.basename(self.model_path)
        if "ridge" in model_filename.lower():
            self.model_name = "ridge"
        elif "lasso" in model_filename.lower():
            self.model_name = "lasso"
        elif "elasticnet" in model_filename.lower():
            self.model_name = "elasticnet"
        elif "random_forest" in model_filename.lower():
            self.model_name = "random_forest"
        elif "gradient" in model_filename.lower():
            self.model_name = "gradient_boosting"
        elif "xgboost" in model_filename.lower():
            self.model_name = "xgboost"
        elif "lightgbm" in model_filename.lower():
            self.model_name = "lightgbm"
        else:
            self.model_name = "best_model"
        
        # Load scaler
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print(f"Loaded scaler from: {self.scaler_path}")
        else:
            print(f"⚠️ Scaler not found at {self.scaler_path}")
        
        # Load metadata
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.model_name = self.metadata.get('model_name', self.model_name)
            self.feature_names = self.metadata.get('features', [])
            print(f"Loaded metadata: {self.model_name}")
        
        print(f"✅ Model loaded successfully: {self.model_name}")
    
    def predict(self, features: Dict[str, float]) -> float:
        """
        Make a single prediction.
        
        Args:
            features: Dictionary mapping feature names to values
            
        Returns:
            Predicted value
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert features to array
        feature_array = self._prepare_features(features)
        
        # Scale if scaler exists
        if self.scaler is not None:
            feature_array = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction = self.model.predict(feature_array)
        
        return float(prediction[0])
    
    def predict_batch(self, instances: List[Dict[str, float]]) -> List[float]:
        """
        Make batch predictions.
        
        Args:
            instances: List of feature dictionaries
            
        Returns:
            List of predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert all instances to array
        feature_arrays = [self._prepare_features(inst) for inst in instances]
        features = np.vstack(feature_arrays)
        
        # Scale if scaler exists
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features)
        
        return [float(p) for p in predictions]
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            numpy array of feature values
        """
        if self.feature_names:
            # Use expected feature order
            try:
                values = [features.get(name, 0.0) for name in self.feature_names]
            except Exception:
                values = list(features.values())
        else:
            # Use provided order
            values = list(features.values())
        
        return np.array(values).reshape(1, -1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__ if self.model else "unknown",
            "features": self.feature_names,
            "metrics": self.metadata.get('metrics', {}),
            "trained_at": self.metadata.get('trained_at', 'unknown'),
            "version": "1.0.0"
        }
