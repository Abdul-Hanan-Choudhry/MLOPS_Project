"""
Unit tests for the data validation module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.validate import (
    DataQualityValidator,
    DataQualityError,
    ValidationResult,
    QualityReport,
    validate_data
)


class TestDataQualityValidator:
    """Tests for DataQualityValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return DataQualityValidator(
            null_threshold=0.01,
            min_rows=10,
            required_columns=["timestamp", "price", "volume", "market_cap"]
        )
    
    @pytest.fixture
    def valid_df(self):
        """Create a valid DataFrame."""
        return pd.DataFrame({
            "timestamp": pd.date_range(start=datetime.utcnow() - timedelta(hours=24), periods=100, freq="H"),
            "price": np.random.uniform(40000, 50000, 100),
            "volume": np.random.uniform(1e9, 2e9, 100),
            "market_cap": np.random.uniform(8e11, 9e11, 100)
        })
    
    @pytest.fixture
    def invalid_df_nulls(self):
        """Create a DataFrame with too many nulls."""
        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="H"),
            "price": np.random.uniform(40000, 50000, 100),
            "volume": np.random.uniform(1e9, 2e9, 100),
            "market_cap": np.random.uniform(8e11, 9e11, 100)
        })
        # Add 5% nulls (exceeds 1% threshold)
        df.loc[:4, "price"] = None
        return df
    
    def test_check_schema_pass(self, validator, valid_df):
        """Test schema validation passes with all required columns."""
        result = validator.check_schema(valid_df)
        assert result.passed is True
        assert result.check_name == "schema_validation"
    
    def test_check_schema_fail(self, validator):
        """Test schema validation fails with missing columns."""
        df = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "price": [100, 200, 300]
            # Missing: volume, market_cap
        })
        result = validator.check_schema(df)
        assert result.passed is False
        assert "volume" in result.details["missing_columns"]
        assert "market_cap" in result.details["missing_columns"]
    
    def test_check_null_values_pass(self, validator, valid_df):
        """Test null check passes with no nulls."""
        result = validator.check_null_values(valid_df)
        assert result.passed is True
    
    def test_check_null_values_fail(self, validator, invalid_df_nulls):
        """Test null check fails when threshold exceeded."""
        result = validator.check_null_values(invalid_df_nulls)
        assert result.passed is False
        assert "price" in result.details["failed_columns"]
    
    def test_check_row_count_pass(self, validator, valid_df):
        """Test row count check passes with sufficient rows."""
        result = validator.check_row_count(valid_df)
        assert result.passed is True
        assert result.details["row_count"] == 100
    
    def test_check_row_count_fail(self, validator):
        """Test row count check fails with insufficient rows."""
        df = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "price": [100, 200, 300],
            "volume": [1e9, 1e9, 1e9],
            "market_cap": [8e11, 8e11, 8e11]
        })
        result = validator.check_row_count(df)
        assert result.passed is False
        assert result.details["row_count"] == 3
    
    def test_check_value_ranges_pass(self, validator, valid_df):
        """Test value range check passes with valid ranges."""
        result = validator.check_value_ranges(valid_df)
        assert result.passed is True
    
    def test_check_value_ranges_fail(self, validator):
        """Test value range check fails with negative values."""
        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="H"),
            "price": [-100, 45000, 45000, 45000, 45000, 45000, 45000, 45000, 45000, 45000],
            "volume": [1e9] * 10,
            "market_cap": [8e11] * 10
        })
        result = validator.check_value_ranges(df)
        assert result.passed is False
    
    def test_validate_overall_pass(self, validator, valid_df):
        """Test overall validation passes."""
        report = validator.validate(valid_df)
        assert report.passed is True
        assert report.passed_checks > 0
    
    def test_validate_or_fail_raises(self, validator, invalid_df_nulls):
        """Test validate_or_fail raises exception on failure."""
        with pytest.raises(DataQualityError):
            validator.validate_or_fail(invalid_df_nulls)
    
    def test_quality_report_to_dict(self, validator, valid_df):
        """Test QualityReport serialization."""
        report = validator.validate(valid_df)
        report_dict = report.to_dict()
        
        assert "passed" in report_dict
        assert "timestamp" in report_dict
        assert "results" in report_dict
        assert isinstance(report_dict["results"], list)


class TestValidateDataFunction:
    """Tests for the validate_data function."""
    
    def test_validate_data_pass(self):
        """Test validate_data function with valid data."""
        df = pd.DataFrame({
            "timestamp": pd.date_range(start=datetime.utcnow() - timedelta(hours=24), periods=200, freq="H"),
            "price": np.random.uniform(40000, 50000, 200),
            "volume": np.random.uniform(1e9, 2e9, 200),
            "market_cap": np.random.uniform(8e11, 9e11, 200)
        })
        
        report = validate_data(df, fail_on_error=False)
        assert report.passed is True
    
    def test_validate_data_fail_raises(self):
        """Test validate_data raises on failure when fail_on_error=True."""
        df = pd.DataFrame({
            "timestamp": [1, 2, 3],  # Too few rows
            "price": [100, 200, 300],
            "volume": [1e9, 1e9, 1e9],
            "market_cap": [8e11, 8e11, 8e11]
        })
        
        with pytest.raises(DataQualityError):
            validate_data(df, fail_on_error=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
