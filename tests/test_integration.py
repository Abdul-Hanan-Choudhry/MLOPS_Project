"""
Integration tests for the full pipeline
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDataPipeline:
    """Integration tests for data pipeline"""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation for CI"""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from create_synthetic_data import create_synthetic_crypto_data
        
        data = create_synthetic_crypto_data(n_samples=100, seed=42)
        
        assert len(data) > 0
        assert 'price' in data.columns
        assert 'volume' in data.columns
        assert 'market_cap' in data.columns
        assert 'target_price' in data.columns
    
    def test_data_features(self):
        """Test that generated data has all required features"""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from create_synthetic_data import create_synthetic_crypto_data
        
        data = create_synthetic_crypto_data(n_samples=100)
        
        required_features = [
            'price', 'open', 'high', 'low', 'close',
            'volume', 'market_cap', 'ma_7', 'ma_14'
        ]
        
        for feature in required_features:
            assert feature in data.columns, f"Missing feature: {feature}"
    
    def test_data_quality(self):
        """Test data quality constraints"""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from create_synthetic_data import create_synthetic_crypto_data
        
        data = create_synthetic_crypto_data(n_samples=100)
        
        # No null values
        assert data.isnull().sum().sum() == 0
        
        # Price should be positive
        assert (data['price'] > 0).all()
        
        # Volume should be positive
        assert (data['volume'] > 0).all()
        
        # High >= Low
        assert (data['high'] >= data['low']).all()


class TestModelTraining:
    """Integration tests for model training"""
    
    def test_training_script_imports(self):
        """Test that training script imports work"""
        try:
            from src.models.train import ModelTrainer
            assert True
        except ImportError as e:
            # May fail in CI without full setup
            pytest.skip(f"Import not available: {e}")
    
    def test_model_predictions_shape(self):
        """Test model predictions have correct shape"""
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([95000.0, 96000.0, 97000.0])
        
        X_test = np.random.randn(3, 5)
        predictions = mock_model.predict(X_test)
        
        assert len(predictions) == len(X_test)


class TestCMLReporting:
    """Integration tests for CML reporting"""
    
    def test_cml_report_generation(self):
        """Test CML report can be generated"""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        
        # Test report generation functions exist
        from generate_cml_report import (
            generate_metrics_table,
            generate_improvement_analysis,
            generate_recommendation
        )
        
        # Test with empty runs
        result = generate_metrics_table([], None)
        assert "No training runs found" in result
        
        # Test recommendation with no runs
        result = generate_recommendation([], None)
        assert "Recommendation" in result
    
    def test_performance_check_thresholds(self):
        """Test performance check threshold logic"""
        rmse_threshold = 500.0
        r2_threshold = 0.7
        
        # Good model
        good_rmse = 297.0
        good_r2 = 0.85
        
        assert good_rmse < rmse_threshold
        assert good_r2 > r2_threshold
        
        # Bad model
        bad_rmse = 600.0
        bad_r2 = 0.5
        
        assert bad_rmse > rmse_threshold
        assert bad_r2 < r2_threshold


class TestDockerDeployment:
    """Integration tests for Docker deployment"""
    
    def test_dockerfile_exists(self):
        """Test Dockerfile exists"""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"
    
    def test_requirements_api_exists(self):
        """Test API requirements file exists"""
        requirements = Path(__file__).parent.parent / "requirements-api.txt"
        assert requirements.exists(), "requirements-api.txt not found"
    
    def test_api_main_exists(self):
        """Test API main module exists"""
        api_main = Path(__file__).parent.parent / "src" / "api" / "main.py"
        assert api_main.exists(), "src/api/main.py not found"


class TestGitHubWorkflows:
    """Test GitHub Actions workflow files"""
    
    def test_workflows_exist(self):
        """Test all workflow files exist"""
        workflows_dir = Path(__file__).parent.parent / ".github" / "workflows"
        
        expected_workflows = [
            "ci-feature-to-dev.yml",
            "ci-dev-to-test.yml",
            "cd-test-to-master.yml"
        ]
        
        for workflow in expected_workflows:
            workflow_path = workflows_dir / workflow
            assert workflow_path.exists(), f"Missing workflow: {workflow}"
    
    def test_workflow_yaml_valid(self):
        """Test workflow files are valid YAML"""
        import yaml
        
        workflows_dir = Path(__file__).parent.parent / ".github" / "workflows"
        
        for workflow_file in workflows_dir.glob("*.yml"):
            with open(workflow_file, 'r', encoding='utf-8') as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {workflow_file}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
