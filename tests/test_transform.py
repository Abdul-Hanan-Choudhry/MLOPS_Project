"""
Unit tests for the feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transform import CryptoFeatureEngineer, transform_data


class TestCryptoFeatureEngineer:
    """Tests for CryptoFeatureEngineer class."""
    
    @pytest.fixture
    def engineer(self):
        """Create a feature engineer instance."""
        return CryptoFeatureEngineer(
            lag_periods=[1, 3, 6],
            rolling_windows=[6, 12],
            target_horizon=1
        )
    
    @pytest.fixture
    def raw_df(self):
        """Create raw DataFrame for testing."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
        return pd.DataFrame({
            "timestamp": dates,
            "price": 45000 + np.cumsum(np.random.randn(100) * 100),
            "volume": np.random.uniform(1e9, 2e9, 100),
            "market_cap": np.random.uniform(8e11, 9e11, 100)
        })
    
    def test_initialization(self, engineer):
        """Test feature engineer initialization."""
        assert engineer.lag_periods == [1, 3, 6]
        assert engineer.rolling_windows == [6, 12]
        assert engineer.target_horizon == 1
    
    def test_create_lag_features(self, engineer, raw_df):
        """Test lag feature creation."""
        df = engineer.create_lag_features(raw_df)
        
        # Check lag features exist
        assert "price_lag_1h" in df.columns
        assert "price_lag_3h" in df.columns
        assert "price_lag_6h" in df.columns
        assert "volume_lag_1h" in df.columns
        
        # Check price change features
        assert "price_change_1h" in df.columns
        assert "price_pct_change_1h" in df.columns
    
    def test_create_rolling_features(self, engineer, raw_df):
        """Test rolling statistics creation."""
        df = engineer.create_rolling_features(raw_df)
        
        # Check rolling mean features
        assert "price_rolling_mean_6h" in df.columns
        assert "price_rolling_mean_12h" in df.columns
        
        # Check rolling std features
        assert "price_rolling_std_6h" in df.columns
        
        # Check volume rolling features
        assert "volume_rolling_mean_6h" in df.columns
    
    def test_create_time_features(self, engineer, raw_df):
        """Test time-based feature creation."""
        df = engineer.create_time_features(raw_df)
        
        # Check basic time features
        assert "hour" in df.columns
        assert "day_of_week" in df.columns
        assert "is_weekend" in df.columns
        
        # Check cyclical features
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        
        # Verify values are in expected ranges
        assert df["hour"].min() >= 0
        assert df["hour"].max() <= 23
        assert df["is_weekend"].isin([0, 1]).all()
    
    def test_create_technical_indicators(self, engineer, raw_df):
        """Test technical indicator creation."""
        # Need rolling features first
        df = engineer.create_rolling_features(raw_df)
        df = engineer.create_lag_features(df)
        df = engineer.create_technical_indicators(df)
        
        # Check volatility features (based on configured rolling windows)
        assert "volatility_6h" in df.columns
        assert "volatility_12h" in df.columns
        
        # Note: momentum features depend on lag_periods configuration
        # With lag_periods=[1,3,6], max_lag=6 < 12, so no momentum features
    
    def test_create_target(self, engineer, raw_df):
        """Test target variable creation."""
        df = engineer.create_target(raw_df)
        
        # Check target columns
        assert "target_price_1h" in df.columns
        assert "target_pct_change_1h" in df.columns
        assert "target_direction_1h" in df.columns
        
        # Verify target is shifted correctly
        assert pd.isna(df["target_price_1h"].iloc[-1])
    
    def test_transform_full_pipeline(self, engineer, raw_df):
        """Test complete transformation pipeline."""
        df_transformed = engineer.transform(raw_df, drop_na=True)
        
        # Should have more columns than input
        assert len(df_transformed.columns) > len(raw_df.columns)
        
        # Should have fewer rows due to NA dropping
        assert len(df_transformed) < len(raw_df)
        
        # Should have no null values
        assert df_transformed.isnull().sum().sum() == 0
    
    def test_get_feature_columns(self, engineer, raw_df):
        """Test feature column identification."""
        df_transformed = engineer.transform(raw_df)
        feature_cols = engineer.get_feature_columns(df_transformed)
        
        # Should not include metadata columns
        assert "timestamp" not in feature_cols
        assert "extraction_time" not in feature_cols
        
        # Should not include target columns
        target_cols = [c for c in feature_cols if c.startswith("target_")]
        assert len(target_cols) == 0
    
    def test_save_processed_data(self, engineer, raw_df, tmp_path):
        """Test saving processed data."""
        df_transformed = engineer.transform(raw_df)
        filepath = engineer.save_processed_data(df_transformed, output_dir=tmp_path)
        
        assert filepath.exists()
        
        # Verify data can be loaded back
        df_loaded = pd.read_parquet(filepath)
        assert len(df_loaded) == len(df_transformed)


class TestTransformDataFunction:
    """Tests for the transform_data function."""
    
    def test_transform_data_basic(self):
        """Test transform_data function."""
        np.random.seed(42)
        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="H"),
            "price": 45000 + np.cumsum(np.random.randn(100) * 100),
            "volume": np.random.uniform(1e9, 2e9, 100),
            "market_cap": np.random.uniform(8e11, 9e11, 100)
        })
        
        df_transformed, data_path, report_path = transform_data(
            df, 
            save=False, 
            generate_report=False
        )
        
        assert isinstance(df_transformed, pd.DataFrame)
        assert len(df_transformed.columns) > len(df.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
