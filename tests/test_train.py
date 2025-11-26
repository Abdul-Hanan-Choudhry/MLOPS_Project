"""
Unit tests for the Model Training Module.
Tests training, evaluation, and registry functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCryptoModelTrainer:
    """Tests for CryptoModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200
        
        # Generate synthetic price data
        prices = 50000 + np.cumsum(np.random.randn(n_samples) * 100)
        
        # Create DataFrame with features
        df = pd.DataFrame({
            'price': prices,
            'volume': np.random.uniform(1e9, 5e9, n_samples),
            'market_cap': prices * 19e6,
            'price_lag_1': np.roll(prices, 1),
            'price_lag_3': np.roll(prices, 3),
            'price_rolling_mean_6': pd.Series(prices).rolling(6).mean().values,
            'price_rolling_std_6': pd.Series(prices).rolling(6).std().values,
            'volume_rolling_mean_6': np.random.uniform(2e9, 4e9, n_samples),
            'hour_sin': np.sin(2 * np.pi * np.arange(n_samples) % 24 / 24),
            'hour_cos': np.cos(2 * np.pi * np.arange(n_samples) % 24 / 24),
            'target_price_1h': np.roll(prices, -1)  # Target: next hour price
        })
        
        # Handle NaN values from rolling operations
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    @pytest.fixture
    def trainer(self):
        """Create trainer instance with local tracking."""
        with patch.dict(os.environ, {
            'DAGSHUB_USERNAME': '',
            'DAGSHUB_TOKEN': '',
            'MLFLOW_TRACKING_URI': ''
        }):
            from src.models.train import CryptoModelTrainer
            return CryptoModelTrainer(experiment_name="test-experiment")
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer.target_column == "target_price_1h"
        assert trainer.test_size == 0.2
        assert len(trainer.MODEL_CONFIGS) >= 5  # At least 5 base models
    
    def test_prepare_data(self, trainer, sample_data):
        """Test data preparation."""
        X_train, X_test, y_train, y_test, features = trainer.prepare_data(sample_data)
        
        # Check shapes
        assert len(X_train) > len(X_test)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check that target column is not in features
        assert 'target_price_1h' not in features
    
    def test_calculate_metrics(self, trainer):
        """Test metric calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = trainer.calculate_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert 0 <= metrics['r2'] <= 1
        assert metrics['mape'] >= 0
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    def test_train_ridge_model(self, mock_log_model, mock_log_metric, 
                                mock_log_param, mock_start_run, trainer, sample_data):
        """Test training a Ridge regression model."""
        # Setup mock
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        # Prepare data
        X_train, X_test, y_train, y_test, features = trainer.prepare_data(sample_data)
        
        # Train model
        model, metrics, run_id = trainer.train_model(
            model_type="ridge",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=features
        )
        
        # Verify
        assert model is not None
        assert 'rmse' in metrics
        assert run_id == "test-run-id"
        
        # Verify model can make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    def test_train_random_forest(self, mock_log_model, mock_log_metric,
                                  mock_log_param, mock_start_run, trainer, sample_data):
        """Test training Random Forest model."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test-rf-run"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        X_train, X_test, y_train, y_test, features = trainer.prepare_data(sample_data)
        
        model, metrics, run_id = trainer.train_model(
            model_type="random_forest",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=features,
            custom_params={"n_estimators": 10, "max_depth": 3}  # Small for fast test
        )
        
        assert model is not None
        assert hasattr(model, 'feature_importances_')
    
    def test_invalid_model_type(self, trainer, sample_data):
        """Test that invalid model type raises error."""
        X_train, X_test, y_train, y_test, features = trainer.prepare_data(sample_data)
        
        with pytest.raises(ValueError, match="Unknown model type"):
            trainer.train_model(
                model_type="invalid_model",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=features
            )


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        with patch.dict(os.environ, {
            'DAGSHUB_USERNAME': '',
            'DAGSHUB_TOKEN': '',
            'MLFLOW_TRACKING_URI': ''
        }):
            from src.models.evaluate import ModelEvaluator
            return ModelEvaluator()
    
    def test_calculate_metrics(self, evaluator):
        """Test comprehensive metric calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        # Check all expected metrics
        expected_metrics = ['rmse', 'mae', 'r2', 'mape', 'mse', 
                          'mean_error', 'std_error', 'max_error',
                          'median_error', 'directional_accuracy']
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
    
    def test_directional_accuracy(self, evaluator):
        """Test directional accuracy calculation."""
        # Perfect directional prediction
        y_true = np.array([100, 110, 120, 115, 125])  # Up, Up, Down, Up
        y_pred = np.array([100, 115, 125, 110, 130])  # Same directions
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert metrics['directional_accuracy'] == 100.0
    
    def test_r2_score_bounds(self, evaluator):
        """Test R² score is within valid bounds."""
        np.random.seed(42)
        y_true = np.random.uniform(1000, 5000, 100)
        
        # Perfect prediction
        metrics_perfect = evaluator.calculate_metrics(y_true, y_true)
        assert metrics_perfect['r2'] == 1.0
        
        # Good prediction
        y_pred = y_true + np.random.normal(0, 100, 100)
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        assert 0 < metrics['r2'] < 1


class TestModelRegistry:
    """Tests for ModelRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create registry instance."""
        with patch.dict(os.environ, {
            'DAGSHUB_USERNAME': '',
            'DAGSHUB_TOKEN': '',
            'MLFLOW_TRACKING_URI': ''
        }):
            from src.models.registry import ModelRegistry
            return ModelRegistry()
    
    def test_valid_stages(self, registry):
        """Test that valid stages are defined."""
        assert "None" in registry.STAGES
        assert "Staging" in registry.STAGES
        assert "Production" in registry.STAGES
        assert "Archived" in registry.STAGES
    
    def test_invalid_stage_transition(self, registry):
        """Test that invalid stage raises error."""
        with pytest.raises(ValueError, match="Invalid stage"):
            registry.transition_model_stage(
                model_name="test-model",
                version="1",
                stage="InvalidStage"
            )


class TestTrainModelFunction:
    """Tests for the main train_model function."""
    
    @pytest.fixture
    def temp_data_file(self, tmp_path):
        """Create temporary data file."""
        np.random.seed(42)
        n_samples = 100
        prices = 50000 + np.cumsum(np.random.randn(n_samples) * 100)
        
        df = pd.DataFrame({
            'price': prices,
            'volume': np.random.uniform(1e9, 5e9, n_samples),
            'market_cap': prices * 19e6,
            'price_lag_1': np.roll(prices, 1),
            'price_rolling_mean_6': pd.Series(prices).rolling(6).mean().values,
            'target_price_1h': np.roll(prices, -1)
        })
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        filepath = tmp_path / "test_data.parquet"
        df.to_parquet(filepath)
        return filepath
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    @patch('mlflow.log_artifact')
    def test_train_model_function(self, mock_artifact, mock_log_model, 
                                  mock_log_metric, mock_log_param,
                                  mock_start_run, temp_data_file):
        """Test main train_model function."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        with patch.dict(os.environ, {
            'DAGSHUB_USERNAME': '',
            'DAGSHUB_TOKEN': '',
            'MLFLOW_TRACKING_URI': ''
        }):
            from src.models.train import train_model
            
            results = train_model(
                data_path=str(temp_data_file),
                model_types=["ridge"],  # Only Ridge for fast test
                experiment_name="test-experiment",
                save_best=False
            )
        
        assert "ridge" in results
        assert "metrics" in results["ridge"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
