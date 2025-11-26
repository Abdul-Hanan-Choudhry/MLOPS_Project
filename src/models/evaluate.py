"""
Model Evaluation Module
Provides comprehensive evaluation of trained models for Bitcoin price prediction.

This module:
- Evaluates model performance on test data
- Generates evaluation reports
- Creates visualization plots
- Logs evaluation metrics to MLflow
- Supports model comparison
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
import mlflow

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not installed, skipping visualizations")


class ModelEvaluator:
    """
    Evaluates trained machine learning models.
    
    Provides:
    - Comprehensive metrics calculation
    - Performance visualization
    - Model comparison
    - MLflow logging integration
    """
    
    def __init__(
        self,
        tracking_uri: str = None,
        experiment_name: str = "crypto-price-prediction"
    ):
        """
        Initialize the evaluator.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
        """
        self.experiment_name = experiment_name
        self._setup_mlflow(tracking_uri)
        
        logger.info("Initialized ModelEvaluator")
    
    def _setup_mlflow(self, tracking_uri: str = None):
        """Configure MLflow tracking."""
        dagshub_username = os.getenv("DAGSHUB_USERNAME", "")
        dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
        mlflow_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
        
        if dagshub_username and dagshub_token and mlflow_uri:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            mlflow.set_tracking_uri(mlflow_uri)
        else:
            local_uri = f"file://{MODELS_DIR}/mlruns"
            mlflow.set_tracking_uri(local_uri)
        
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}")
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
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
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "mse": mean_squared_error(y_true, y_pred),
        }
        
        # Additional metrics
        errors = y_true - y_pred
        metrics["mean_error"] = np.mean(errors)
        metrics["std_error"] = np.std(errors)
        metrics["max_error"] = np.max(np.abs(errors))
        metrics["median_error"] = np.median(np.abs(errors))
        
        # Directional accuracy (for trading)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics["directional_accuracy"] = np.mean(true_direction == pred_direction) * 100
        
        return metrics
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        log_to_mlflow: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test target
            model_name: Name for logging
            log_to_mlflow: Whether to log to MLflow
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Log results
        logger.info(f"Evaluation Results for {model_name}:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.2f}%")
        
        # Log to MLflow
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"eval_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"eval_{metric_name}", value)
                mlflow.set_tag("evaluation", "true")
                mlflow.set_tag("model_name", model_name)
        
        return metrics
    
    def evaluate_from_file(
        self,
        model_path: str,
        data_path: str = None,
        log_to_mlflow: bool = True
    ) -> Dict[str, float]:
        """
        Load and evaluate a saved model.
        
        Args:
            model_path: Path to saved model pickle
            data_path: Path to test data
            log_to_mlflow: Whether to log to MLflow
            
        Returns:
            Evaluation metrics
        """
        # Load model
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data["model"]
        scaler = model_data.get("scaler")
        model_type = model_data.get("model_type", "unknown")
        
        # Load data
        if data_path is None:
            processed_files = list(Path(PROCESSED_DATA_DIR).glob("processed_crypto_*.parquet"))
            if not processed_files:
                raise FileNotFoundError(f"No processed data in {PROCESSED_DATA_DIR}")
            data_path = max(processed_files, key=lambda x: x.stat().st_mtime)
        
        df = pd.read_parquet(data_path)
        
        # Prepare test data (use last 20%)
        target_col = "target_price_1h"
        df = df.dropna(subset=[target_col])
        
        exclude_cols = [target_col, 'timestamp', 'extraction_time']
        feature_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Use last 20% for test
        split_idx = int(len(X) * 0.8)
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        # Scale if scaler available
        if scaler:
            X_test = scaler.transform(np.nan_to_num(X_test, nan=0.0))
        
        # Evaluate
        return self.evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_type,
            log_to_mlflow=log_to_mlflow
        )
    
    def create_evaluation_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        output_dir: Path = None
    ) -> List[str]:
        """
        Create evaluation visualization plots.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Model name for titles
            output_dir: Output directory for plots
            
        Returns:
            List of plot file paths
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plots")
            return []
        
        output_dir = output_dir or Path(MODELS_DIR) / "plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Actual vs Predicted scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='none')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Price', fontsize=12)
        ax.set_ylabel('Predicted Price', fontsize=12)
        ax.set_title(f'{model_name}: Actual vs Predicted', fontsize=14)
        
        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plot_path = output_dir / f"actual_vs_pred_{model_name}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plot_paths.append(str(plot_path))
        
        # 2. Residuals plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        residuals = y_true - y_pred
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='none')
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Price', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title('Residuals vs Predicted', fontsize=14)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residual Value', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residuals Distribution', fontsize=14)
        
        plot_path = output_dir / f"residuals_{model_name}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plot_paths.append(str(plot_path))
        
        # 3. Time series comparison (if ordered)
        fig, ax = plt.subplots(figsize=(14, 6))
        x_axis = range(len(y_true))
        ax.plot(x_axis, y_true, label='Actual', alpha=0.7)
        ax.plot(x_axis, y_pred, label='Predicted', alpha=0.7)
        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title(f'{model_name}: Price Prediction Over Time', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = output_dir / f"time_series_{model_name}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plot_paths.append(str(plot_path))
        
        # 4. Error distribution by price range
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bin by actual price
        price_bins = pd.cut(y_true, bins=10)
        errors_by_bin = pd.DataFrame({
            'bin': price_bins,
            'error': np.abs(residuals)
        }).groupby('bin')['error'].mean()
        
        errors_by_bin.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Price Range', fontsize=12)
        ax.set_ylabel('Mean Absolute Error', fontsize=12)
        ax.set_title('Error Distribution by Price Range', fontsize=14)
        plt.xticks(rotation=45)
        
        plot_path = output_dir / f"error_by_range_{model_name}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plot_paths.append(str(plot_path))
        
        logger.info(f"Created {len(plot_paths)} evaluation plots in {output_dir}")
        
        return plot_paths
    
    def generate_evaluation_report(
        self,
        metrics: Dict[str, float],
        model_name: str,
        output_dir: Path = None
    ) -> str:
        """
        Generate a text evaluation report.
        
        Args:
            metrics: Evaluation metrics
            model_name: Name of evaluated model
            output_dir: Output directory
            
        Returns:
            Path to report file
        """
        output_dir = output_dir or Path(MODELS_DIR) / "reports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"eval_report_{model_name}_{timestamp}.txt"
        
        report = f"""
================================================================================
MODEL EVALUATION REPORT
================================================================================
Model: {model_name}
Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

PERFORMANCE METRICS
-------------------
Root Mean Square Error (RMSE): {metrics.get('rmse', 'N/A'):.4f}
Mean Absolute Error (MAE):     {metrics.get('mae', 'N/A'):.4f}
R-squared (R²):                {metrics.get('r2', 'N/A'):.4f}
Mean Absolute % Error (MAPE):  {metrics.get('mape', 'N/A'):.2f}%

ADDITIONAL METRICS
------------------
Mean Squared Error (MSE):      {metrics.get('mse', 'N/A'):.4f}
Mean Error (Bias):             {metrics.get('mean_error', 'N/A'):.4f}
Error Std Deviation:           {metrics.get('std_error', 'N/A'):.4f}
Maximum Error:                 {metrics.get('max_error', 'N/A'):.4f}
Median Absolute Error:         {metrics.get('median_error', 'N/A'):.4f}
Directional Accuracy:          {metrics.get('directional_accuracy', 'N/A'):.2f}%

INTERPRETATION
--------------
- RMSE of {metrics.get('rmse', 0):.4f} means predictions are typically off by ~${metrics.get('rmse', 0):.2f}
- R² of {metrics.get('r2', 0):.4f} means the model explains {metrics.get('r2', 0)*100:.1f}% of price variance
- MAPE of {metrics.get('mape', 0):.2f}% indicates average relative prediction error
- Directional accuracy shows how often the model correctly predicts price movement direction

================================================================================
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return str(report_path)
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = "rmse"
    ) -> pd.DataFrame:
        """
        Compare multiple models based on metrics.
        
        Args:
            results: Dictionary of model results
            metric: Primary metric for ranking
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, result in results.items():
            if "metrics" in result:
                row = {"model": model_name}
                row.update(result["metrics"])
                comparison_data.append(row)
        
        if not comparison_data:
            logger.warning("No models to compare")
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric (ascending for error metrics)
        ascending = metric in ["rmse", "mae", "mape", "mse"]
        df = df.sort_values(metric, ascending=ascending)
        
        # Add rank
        df["rank"] = range(1, len(df) + 1)
        
        return df


def evaluate_model(
    model_path: str = None,
    data_path: str = None,
    create_plots: bool = True,
    generate_report: bool = True
) -> Dict[str, Any]:
    """
    Main evaluation function - can be called from Airflow DAG.
    
    Args:
        model_path: Path to saved model
        data_path: Path to test data
        create_plots: Whether to create visualization plots
        generate_report: Whether to generate text report
        
    Returns:
        Evaluation results dictionary
    """
    logger.info("="*60)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("="*60)
    
    # Find model if not specified
    if model_path is None:
        model_files = list(Path(MODELS_DIR).glob("best_model_*.pkl"))
        if not model_files:
            raise FileNotFoundError(f"No saved models found in {MODELS_DIR}")
        model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    scaler = model_data.get("scaler")
    model_type = model_data.get("model_type", "unknown")
    
    # Load and prepare data
    if data_path is None:
        processed_files = list(Path(PROCESSED_DATA_DIR).glob("processed_crypto_*.parquet"))
        if not processed_files:
            raise FileNotFoundError(f"No processed data in {PROCESSED_DATA_DIR}")
        data_path = str(max(processed_files, key=lambda x: x.stat().st_mtime))
    
    df = pd.read_parquet(data_path)
    target_col = "target_price_1h"
    df = df.dropna(subset=[target_col])
    
    exclude_cols = [target_col, 'timestamp', 'extraction_time']
    feature_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    split_idx = int(len(X) * 0.8)
    X_test = np.nan_to_num(X[split_idx:], nan=0.0)
    y_test = y[split_idx:]
    
    if scaler:
        X_test = scaler.transform(X_test)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    
    results = {
        "model_path": model_path,
        "model_type": model_type,
        "metrics": metrics
    }
    
    # Create plots
    if create_plots:
        plot_paths = evaluator.create_evaluation_plots(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_type
        )
        results["plots"] = plot_paths
    
    # Generate report
    if generate_report:
        report_path = evaluator.generate_evaluation_report(
            metrics=metrics,
            model_name=model_type
        )
        results["report"] = report_path
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"evaluation_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"eval_{metric_name}", value)
        
        if create_plots and results.get("plots"):
            for plot_path in results["plots"]:
                mlflow.log_artifact(plot_path, "evaluation_plots")
        
        if generate_report and results.get("report"):
            mlflow.log_artifact(results["report"], "evaluation_reports")
        
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("stage", "evaluation")
    
    logger.info("="*60)
    logger.info("MODEL EVALUATION COMPLETE")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    results = evaluate_model()
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {results['model_type']}")
    print(f"RMSE: {results['metrics']['rmse']:.4f}")
    print(f"MAE: {results['metrics']['mae']:.4f}")
    print(f"R²: {results['metrics']['r2']:.4f}")
    print(f"MAPE: {results['metrics']['mape']:.2f}%")
