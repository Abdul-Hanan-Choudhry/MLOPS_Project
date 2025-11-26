"""
Unit tests for the data extraction module.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.extract import CoinGeckoExtractor, extract_data


class TestCoinGeckoExtractor:
    """Tests for CoinGeckoExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a CoinGeckoExtractor instance."""
        return CoinGeckoExtractor(coin_id="bitcoin", vs_currency="usd")
    
    @pytest.fixture
    def mock_api_response(self):
        """Create mock API response data."""
        return {
            "prices": [
                [1700000000000, 45000.0],
                [1700003600000, 45100.0],
                [1700007200000, 45200.0],
            ],
            "market_caps": [
                [1700000000000, 850000000000.0],
                [1700003600000, 851000000000.0],
                [1700007200000, 852000000000.0],
            ],
            "total_volumes": [
                [1700000000000, 25000000000.0],
                [1700003600000, 25500000000.0],
                [1700007200000, 26000000000.0],
            ]
        }
    
    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.coin_id == "bitcoin"
        assert extractor.vs_currency == "usd"
        assert "coingecko" in extractor.api_url.lower()
    
    @patch('requests.get')
    def test_fetch_market_chart(self, mock_get, extractor, mock_api_response):
        """Test fetching market chart data."""
        mock_get.return_value.json.return_value = mock_api_response
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        
        df = extractor.fetch_market_chart(days=7)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert "volume" in df.columns
        assert "market_cap" in df.columns
        assert "extraction_time" in df.columns
    
    @patch('requests.get')
    def test_fetch_market_chart_error_handling(self, mock_get, extractor):
        """Test error handling for API failures."""
        mock_get.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            extractor.fetch_market_chart(days=7)
    
    def test_save_raw_data(self, extractor, tmp_path):
        """Test saving raw data to disk."""
        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=5, freq="H"),
            "price": [45000, 45100, 45200, 45300, 45400],
            "volume": [1e9] * 5,
            "market_cap": [8e11] * 5
        })
        
        filepath = extractor.save_raw_data(df, output_dir=tmp_path)
        
        assert filepath.exists()
        assert filepath.suffix == ".parquet"
        
        # Verify data can be read back
        df_loaded = pd.read_parquet(filepath)
        assert len(df_loaded) == 5


class TestExtractDataFunction:
    """Tests for the extract_data function."""
    
    @patch('src.data.extract.CoinGeckoExtractor')
    def test_extract_data_with_save(self, MockExtractor, tmp_path):
        """Test extract_data function with save option."""
        mock_df = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=5, freq="H"),
            "price": [45000] * 5,
            "volume": [1e9] * 5,
            "market_cap": [8e11] * 5
        })
        
        mock_instance = MockExtractor.return_value
        mock_instance.fetch_market_chart.return_value = mock_df
        mock_instance.save_raw_data.return_value = tmp_path / "test.parquet"
        
        df, filepath = extract_data(coin_id="bitcoin", days=7, save=True)
        
        assert isinstance(df, pd.DataFrame)
        mock_instance.fetch_market_chart.assert_called_once()
        mock_instance.save_raw_data.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
