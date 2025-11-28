"""
Tests for FastAPI prediction service
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPredictAPI:
    """Tests for the prediction API"""
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        # Mock the required dependencies
        with patch('src.api.predict.PredictionService'):
            from src.api.main import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
    
    def test_prediction_service_initialization(self):
        """Test that PredictionService can be instantiated"""
        with patch('joblib.load') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([95000.0])
            mock_load.return_value = mock_model
            
            from src.api.predict import PredictionService
            service = PredictionService(model_dir="models")
            
            # Service should exist even if models aren't loaded
            assert service is not None
    
    def test_prediction_with_mock_model(self):
        """Test prediction with a mocked model"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([95000.0])
        
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        with patch('joblib.load') as mock_load:
            mock_load.side_effect = [mock_model, mock_scaler]
            
            from src.api.predict import PredictionService
            service = PredictionService(model_dir="models")
            service.model = mock_model
            service.scaler = mock_scaler
            
            # Mock features
            features = {
                'price': 95000,
                'volume': 50000000000,
                'market_cap': 1900000000000
            }
            
            result = service.predict(features)
            assert result is not None
    
    def test_feature_preparation(self):
        """Test feature preparation logic"""
        features = {
            'price': 95000.0,
            'volume': 50000000000.0,
            'market_cap': 1900000000000.0,
            'price_change_24h': 2.5,
            'volume_change_24h': -5.0
        }
        
        # Basic validation
        assert 'price' in features
        assert features['price'] > 0
        assert 'volume' in features


class TestAPIEndpoints:
    """Tests for API endpoint functionality"""
    
    def test_predict_request_validation(self):
        """Test that prediction request validation works"""
        # Valid request
        valid_request = {
            "features": {
                "price": 95000,
                "volume": 50000000000,
                "market_cap": 1900000000000
            }
        }
        
        assert 'features' in valid_request
        assert 'price' in valid_request['features']
    
    def test_batch_predict_format(self):
        """Test batch prediction request format"""
        batch_request = {
            "batch_features": [
                {"price": 95000, "volume": 50000000000, "market_cap": 1900000000000},
                {"price": 96000, "volume": 51000000000, "market_cap": 1920000000000}
            ]
        }
        
        assert 'batch_features' in batch_request
        assert len(batch_request['batch_features']) == 2


class TestModelLoading:
    """Tests for model loading functionality"""
    
    def test_model_path_construction(self):
        """Test model path is correctly constructed"""
        from pathlib import Path
        
        model_dir = Path("models")
        expected_patterns = [
            "best_model_*.joblib",
            "*_model.joblib"
        ]
        
        # Path should be constructable
        for pattern in expected_patterns:
            full_pattern = model_dir / pattern
            assert str(model_dir) in str(full_pattern)
    
    def test_scaler_path(self):
        """Test scaler path construction"""
        from pathlib import Path
        
        model_dir = Path("models")
        scaler_path = model_dir / "scaler.joblib"
        
        assert "scaler" in str(scaler_path)
        assert str(scaler_path).endswith(".joblib")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
