"""
Model Training Module with MLflow Tracking
Trains ML models for Bitcoin price prediction with comprehensive experiment tracking.

This module:
- Trains multiple model types (Ridge, RandomForest, XGBoost, LightGBM)
- Logs all hyperparameters to MLflow
- Logs metrics (RMSE, MAE, R², MAPE)
- Logs trained model as artifact
- Supports hyperparameter tuning
- Integrates with DagsHub MLflow server
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed, skipping XGBoost models")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed, skipping LightGBM models")


class CryptoModelTrainer:
    """
    Trains machine learning models for cryptocurrency price prediction.
    
    Integrates with MLflow for comprehensive experiment tracking including:
    - Hyperparameters logging
    - Metrics logging (RMSE, MAE, R², MAPE)
    - Model artifact logging
    - Feature importance logging
    """
    
    # Model configurations with default hyperparameters
    MODEL_CONFIGS = {
        "ridge": {
            "class": Ridge,
            "params": {
                "alpha": 1.0,
                "fit_intercept": True,
                "solver": "auto"
            }
        },
        "lasso": {
            "class": Lasso,
            "params": {
                "alpha": 0.1,
                "fit_intercept": True,
                "max_iter": 1000
            }
        },
        "elasticnet": {
            "class": ElasticNet,
            "params": {
                "alpha": 0.1,
                "l1_ratio": 0.5,
                "fit_intercept": True,
                "max_iter": 1000
            }
        },
        "random_forest": {
            "class": RandomForestRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        "gradient_boosting": {
            "class": GradientBoostingRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            }
        }
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        MODEL_CONFIGS["xgboost"] = {
            "class": xgb.XGBRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0
            }
        }
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        MODEL_CONFIGS["lightgbm"] = {
            "class": lgb.LGBMRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            }
        }
    
    def __init__(
        self,
        experiment_name: str = "crypto-price-prediction",
        tracking_uri: str = None,
        target_column: str = "target_price_1h",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the model trainer.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI (DagsHub)
            target_column: Name of target variable column
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        self.experiment_name = experiment_name
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Setup MLflow tracking
        self._setup_mlflow(tracking_uri)
        
        logger.info(f"Initialized CryptoModelTrainer")
        logger.info(f"  Experiment: {experiment_name}")
        logger.info(f"  Target: {target_column}")
        logger.info(f"  Available models: {list(self.MODEL_CONFIGS.keys())}")
    
    def _setup_mlflow(self, tracking_uri: str = None):
        """Configure MLflow with DagsHub tracking."""
        # Get credentials from environment
        dagshub_username = os.getenv("DAGSHUB_USERNAME", "")
        dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
        mlflow_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
        
        if dagshub_username and dagshub_token and mlflow_uri:
            # Set authentication
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            
            mlflow.set_tracking_uri(mlflow_uri)
            logger.info(f"MLflow tracking URI: {mlflow_uri}")
        else:
            # Use local tracking
            local_uri = f"file://{MODELS_DIR}/mlruns"
            mlflow.set_tracking_uri(local_uri)
            logger.warning(f"DagsHub not configured, using local tracking: {local_uri}")
        
        # Set or create experiment
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}")
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load processed data for training.
        
        Args:
            data_path: Path to processed parquet file
            
        Returns:
            DataFrame with features and target
        """
        if data_path is None:
            # Find latest processed file
            processed_files = list(Path(PROCESSED_DATA_DIR).glob("processed_crypto_*.parquet"))
            if not processed_files:
                raise FileNotFoundError(f"No processed data found in {PROCESSED_DATA_DIR}")
            data_path = max(processed_files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def prepare_data(
        self, 
        df: pd.DataFrame,
        feature_columns: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature columns to use
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Drop rows with missing target
        df = df.dropna(subset=[self.target_column])
        
        # Select features
        if feature_columns is None:
            # Use all numeric columns except target and timestamp
            exclude_cols = [self.target_column, 'timestamp', 'extraction_time']
            feature_columns = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols
            ]
        
        # Check for missing features
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns].values
        y = df[self.target_column].values
        
        # Handle any remaining NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Split data (time-series aware - no shuffle)
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        logger.info(f"Prepared data: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        return metrics
    
    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        custom_params: Dict[str, Any] = None,
        run_name: str = None
    ) -> Tuple[Any, Dict[str, float], str]:
        """
        Train a single model with MLflow tracking.
        
        Args:
            model_type: Type of model to train
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: List of feature names
            custom_params: Custom hyperparameters (overrides defaults)
            run_name: Name for MLflow run
            
        Returns:
            Trained model, metrics dict, run_id
        """
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[model_type]
        model_class = config["class"]
        params = config["params"].copy()
        
        # Override with custom params
        if custom_params:
            params.update(custom_params)
        
        run_name = run_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Training {model_type} model...")
        
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("target_column", self.target_column)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("random_state", self.random_state)
            
            # Log all hyperparameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Train model
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            test_metrics = self.calculate_metrics(y_test, y_test_pred)
            
            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Log cross-validation score
            try:
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(
                    model_class(**params), 
                    np.vstack([X_train, X_test]),
                    np.concatenate([y_train, y_test]),
                    cv=tscv,
                    scoring='neg_root_mean_squared_error'
                )
                mlflow.log_metric("cv_rmse_mean", -cv_scores.mean())
                mlflow.log_metric("cv_rmse_std", cv_scores.std())
            except Exception as e:
                logger.warning(f"Could not compute CV scores: {e}")
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save and log feature importance
                importance_path = f"/tmp/feature_importance_{model_type}.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path, "feature_importance")
                
                # Log top 10 features as params
                for i, row in importance_df.head(10).iterrows():
                    mlflow.log_param(f"top_feature_{importance_df.index.get_loc(i)+1}", row['feature'])
            
            elif hasattr(model, 'coef_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': np.abs(model.coef_)
                }).sort_values('coefficient', ascending=False)
                
                importance_path = f"/tmp/feature_coefficients_{model_type}.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path, "feature_importance")
            
            # Log model with signature
            signature = infer_signature(X_train, y_train_pred)
            mlflow.sklearn.log_model(
                model, 
                "model",
                signature=signature,
                registered_model_name=f"crypto-{model_type}"
            )
            
            # Log scaler
            scaler_path = "/tmp/scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            mlflow.log_artifact(scaler_path, "preprocessing")
            
            # Log feature names
            features_path = "/tmp/feature_names.json"
            with open(features_path, 'w') as f:
                json.dump(feature_names, f)
            mlflow.log_artifact(features_path, "preprocessing")
            
            # Set tags
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("stage", "training")
            mlflow.set_tag("dataset", "bitcoin")
            
            logger.info(f"✓ {model_type} training complete")
            logger.info(f"  Test RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"  Test MAE: {test_metrics['mae']:.4f}")
            logger.info(f"  Test R²: {test_metrics['r2']:.4f}")
            logger.info(f"  Test MAPE: {test_metrics['mape']:.2f}%")
            logger.info(f"  Run ID: {run_id}")
            
            return model, test_metrics, run_id
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        models_to_train: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models and compare.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: Feature names
            models_to_train: List of model types to train (default: all)
            
        Returns:
            Dictionary of results for each model
        """
        models_to_train = models_to_train or list(self.MODEL_CONFIGS.keys())
        
        results = {}
        best_model = None
        best_rmse = float('inf')
        
        logger.info(f"Training {len(models_to_train)} models...")
        
        for model_type in models_to_train:
            try:
                model, metrics, run_id = self.train_model(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    feature_names=feature_names
                )
                
                results[model_type] = {
                    "model": model,
                    "metrics": metrics,
                    "run_id": run_id
                }
                
                # Track best model
                if metrics['rmse'] < best_rmse:
                    best_rmse = metrics['rmse']
                    best_model = model_type
                    
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                results[model_type] = {"error": str(e)}
        
        # Log comparison summary
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'MAPE':<12}")
        logger.info("-"*60)
        
        for model_type, result in results.items():
            if "metrics" in result:
                m = result["metrics"]
                logger.info(f"{model_type:<20} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f} {m['mape']:<12.2f}%")
        
        logger.info("-"*60)
        logger.info(f"Best Model: {best_model} (RMSE: {best_rmse:.4f})")
        logger.info("="*60 + "\n")
        
        return results
    
    def save_best_model(
        self,
        results: Dict[str, Dict[str, Any]],
        output_dir: Path = None
    ) -> Path:
        """
        Save the best performing model locally.
        
        Args:
            results: Training results from train_all_models
            output_dir: Output directory
            
        Returns:
            Path to saved model
        """
        output_dir = output_dir or MODELS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find best model
        best_model = None
        best_rmse = float('inf')
        best_type = None
        
        for model_type, result in results.items():
            if "metrics" in result and result["metrics"]["rmse"] < best_rmse:
                best_rmse = result["metrics"]["rmse"]
                best_model = result["model"]
                best_type = model_type
        
        if best_model is None:
            raise ValueError("No successful model found")
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"best_model_{best_type}_{timestamp}.pkl"
        model_path = output_dir / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                "model": best_model,
                "model_type": best_type,
                "scaler": self.scaler,
                "metrics": results[best_type]["metrics"],
                "timestamp": timestamp
            }, f)
        
        logger.info(f"Saved best model ({best_type}) to {model_path}")
        
        return model_path


def train_model(
    data_path: str = None,
    model_types: List[str] = None,
    experiment_name: str = "crypto-price-prediction",
    save_best: bool = True
) -> Dict[str, Any]:
    """
    Main training function - can be called from Airflow DAG.
    
    Args:
        data_path: Path to processed data file
        model_types: List of model types to train
        experiment_name: MLflow experiment name
        save_best: Whether to save best model locally
        
    Returns:
        Training results dictionary
    """
    logger.info("="*60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("="*60)
    
    # Initialize trainer
    trainer = CryptoModelTrainer(experiment_name=experiment_name)
    
    # Load data
    df = trainer.load_data(data_path)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(df)
    
    # Train models
    results = trainer.train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        models_to_train=model_types
    )
    
    # Save best model
    if save_best:
        model_path = trainer.save_best_model(results)
        results["best_model_path"] = str(model_path)
    
    logger.info("="*60)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    # Run training when executed directly
    results = train_model()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    for model_type, result in results.items():
        if model_type != "best_model_path" and "metrics" in result:
            print(f"\n{model_type}:")
            print(f"  RMSE: {result['metrics']['rmse']:.4f}")
            print(f"  MAE: {result['metrics']['mae']:.4f}")
            print(f"  R²: {result['metrics']['r2']:.4f}")
            print(f"  MAPE: {result['metrics']['mape']:.2f}%")
