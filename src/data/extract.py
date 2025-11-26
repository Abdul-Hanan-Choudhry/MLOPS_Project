"""
CoinGecko Data Extractor Module
Handles fetching cryptocurrency data from CoinGecko API.

Supports:
- CoinGecko API (with optional API key)
- Synthetic data generation for testing/development
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
import requests

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import coingecko_config, RAW_DATA_DIR

logger = get_logger(__name__)


def generate_synthetic_crypto_data(
    coin_id: str = "bitcoin",
    days: int = 30,
    base_price: float = 95000.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """
    Generate synthetic cryptocurrency data for testing/development.
    
    This is useful when:
    - CoinGecko API is unavailable or rate-limited
    - Testing the pipeline without API calls
    - Development and debugging
    
    Args:
        coin_id: Cryptocurrency ID for metadata
        days: Number of days of data to generate
        base_price: Starting price
        volatility: Price volatility factor
        
    Returns:
        DataFrame mimicking CoinGecko API response
    """
    logger.info(f"Generating {days} days of synthetic data for {coin_id}")
    
    # Generate hourly timestamps
    hours = days * 24
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    timestamps = pd.date_range(start=start_time, end=end_time, freq="H")
    
    # Generate realistic price movements using geometric Brownian motion
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.0001, volatility, len(timestamps))
    price_series = base_price * np.exp(np.cumsum(returns))
    
    # Generate correlated volume and market cap
    base_volume = 50e9  # $50B daily volume
    base_market_cap = base_price * 19.5e6  # ~19.5M BTC supply
    
    volume_noise = np.random.uniform(0.7, 1.3, len(timestamps))
    volumes = base_volume / 24 * volume_noise  # Hourly volume
    
    market_caps = price_series * 19.5e6  # Market cap tracks price
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": price_series,
        "market_cap": market_caps,
        "volume": volumes,
        "extraction_time": datetime.utcnow(),
        "coin_id": coin_id,
        "vs_currency": "usd"
    })
    
    logger.info(f"Generated {len(df)} synthetic records")
    
    return df


class CoinGeckoExtractor:
    """
    Extracts cryptocurrency market data from CoinGecko API.
    
    CoinGecko provides free access to crypto market data including:
    - Historical prices
    - Market cap
    - Trading volume
    
    Note: Uses the demo API endpoint for free access without API key.
    For production, obtain an API key from https://www.coingecko.com/en/api
    """
    
    # Demo API URL (free, no API key required)
    DEMO_API_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(
        self,
        coin_id: str = None,
        vs_currency: str = None,
        api_url: str = None,
        api_key: str = None
    ):
        """
        Initialize the CoinGecko extractor.
        
        Args:
            coin_id: Cryptocurrency ID (e.g., 'bitcoin', 'ethereum')
            vs_currency: Target currency for prices (e.g., 'usd', 'eur')
            api_url: CoinGecko API base URL
            api_key: Optional API key for higher rate limits
        """
        self.coin_id = coin_id or coingecko_config.COIN_ID
        self.vs_currency = vs_currency or coingecko_config.VS_CURRENCY
        self.api_url = api_url or coingecko_config.API_URL
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY")
        
        logger.info(f"Initialized CoinGeckoExtractor for {self.coin_id}/{self.vs_currency}")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """
        Make a request to the CoinGecko API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.api_url}/{endpoint}"
        
        # Add API key header if available
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        
        logger.debug(f"Making request to {url} with params {params}")
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            logger.debug(f"Received response with status {response.status_code}")
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Request to {url} timed out")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from {url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def fetch_market_chart(self, days: int = None) -> pd.DataFrame:
        """
        Fetch historical market data for the cryptocurrency.
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with columns: timestamp, price, market_cap, volume
        """
        days = days or coingecko_config.DATA_FETCH_DAYS
        
        logger.info(f"Fetching {days} days of market data for {self.coin_id}")
        
        endpoint = f"coins/{self.coin_id}/market_chart"
        params = {
            "vs_currency": self.vs_currency,
            "days": days,
            "interval": "hourly" if days <= 90 else "daily"
        }
        
        data = self._make_request(endpoint, params)
        
        # Parse the response
        df = pd.DataFrame({
            "timestamp": [x[0] for x in data["prices"]],
            "price": [x[1] for x in data["prices"]],
            "market_cap": [x[1] for x in data["market_caps"]],
            "volume": [x[1] for x in data["total_volumes"]]
        })
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Add extraction metadata
        df["extraction_time"] = datetime.utcnow()
        df["coin_id"] = self.coin_id
        df["vs_currency"] = self.vs_currency
        
        logger.info(f"Fetched {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def fetch_current_price(self) -> Dict[str, float]:
        """
        Fetch the current price of the cryptocurrency.
        
        Returns:
            Dictionary with current price information
        """
        logger.info(f"Fetching current price for {self.coin_id}")
        
        endpoint = "simple/price"
        params = {
            "ids": self.coin_id,
            "vs_currencies": self.vs_currency,
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }
        
        data = self._make_request(endpoint, params)
        return data.get(self.coin_id, {})
    
    def fetch_coin_info(self) -> Dict[str, Any]:
        """
        Fetch detailed information about the cryptocurrency.
        
        Returns:
            Dictionary with coin metadata
        """
        logger.info(f"Fetching coin info for {self.coin_id}")
        
        endpoint = f"coins/{self.coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false"
        }
        
        return self._make_request(endpoint, params)
    
    def save_raw_data(
        self,
        df: pd.DataFrame,
        output_dir: Path = None,
        file_format: str = "parquet"
    ) -> Path:
        """
        Save raw data to disk with timestamp.
        
        Args:
            df: DataFrame to save
            output_dir: Output directory (defaults to RAW_DATA_DIR)
            file_format: File format ('parquet' or 'csv')
            
        Returns:
            Path to the saved file
        """
        output_dir = output_dir or RAW_DATA_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_{self.coin_id}_{timestamp}.{file_format}"
        filepath = output_dir / filename
        
        # Save based on format
        if file_format == "parquet":
            df.to_parquet(filepath, index=False)
        elif file_format == "csv":
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Saved raw data to {filepath}")
        
        return filepath


def extract_data(
    coin_id: str = None,
    days: int = None,
    output_dir: Path = None,
    save: bool = True,
    use_synthetic: bool = False
) -> tuple[pd.DataFrame, Optional[Path]]:
    """
    Main extraction function to fetch and optionally save crypto data.
    
    Args:
        coin_id: Cryptocurrency ID
        days: Number of days to fetch
        output_dir: Output directory for raw data
        save: Whether to save the data to disk
        use_synthetic: Use synthetic data if API fails or for testing
        
    Returns:
        Tuple of (DataFrame, filepath or None)
    """
    coin_id = coin_id or coingecko_config.COIN_ID
    days = days or coingecko_config.DATA_FETCH_DAYS
    
    extractor = CoinGeckoExtractor(coin_id=coin_id)
    
    df = None
    
    # Try to fetch from API first (unless synthetic is explicitly requested)
    if not use_synthetic:
        try:
            df = extractor.fetch_market_chart(days=days)
        except requests.exceptions.HTTPError as e:
            if "401" in str(e) or "429" in str(e):
                logger.warning(f"API request failed ({e}). Falling back to synthetic data.")
                use_synthetic = True
            else:
                raise
        except Exception as e:
            logger.warning(f"API request failed: {e}. Falling back to synthetic data.")
            use_synthetic = True
    
    # Generate synthetic data if needed
    if use_synthetic or df is None:
        df = generate_synthetic_crypto_data(coin_id=coin_id, days=days)
    
    filepath = None
    if save:
        filepath = extractor.save_raw_data(df, output_dir=output_dir)
    
    return df, filepath


if __name__ == "__main__":
    # Test the extractor
    logger.info("Running CoinGecko extractor test...")
    
    df, filepath = extract_data(coin_id="bitcoin", days=7, save=True)
    
    print(f"\n{'='*50}")
    print("Extraction Results")
    print(f"{'='*50}")
    print(f"Records fetched: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Saved to: {filepath}")
    print(f"\nSample data:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
