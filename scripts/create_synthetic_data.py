"""
Create Synthetic Training Data for CI Pipeline
Generates realistic crypto market data for testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def create_synthetic_crypto_data(n_samples=1000, seed=42):
    """
    Generate synthetic cryptocurrency market data
    
    Args:
        n_samples: Number of data points to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic crypto data
    """
    np.random.seed(seed)
    
    # Base values (realistic for Bitcoin)
    base_price = 95000
    base_volume = 50_000_000_000
    base_market_cap = 1_900_000_000_000
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_samples)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate realistic price movements (random walk with drift)
    returns = np.random.normal(0.0002, 0.02, n_samples)  # Daily returns
    price_multiplier = np.cumprod(1 + returns)
    prices = base_price * price_multiplier
    
    # Add some volatility clustering
    volatility = np.abs(np.random.normal(0, 0.01, n_samples))
    prices = prices * (1 + volatility * np.sign(np.random.randn(n_samples)))
    
    # Generate correlated volume (higher on big moves)
    volume_noise = np.random.lognormal(0, 0.3, n_samples)
    price_changes = np.abs(np.diff(prices, prepend=prices[0]))
    volume = base_volume * volume_noise * (1 + price_changes / prices * 10)
    
    # Market cap follows price
    market_cap = prices * 19_500_000  # Approximate BTC supply
    
    # Generate OHLCV data
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
    open_price = np.roll(prices, 1)
    open_price[0] = prices[0]
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'open': open_price,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
        'market_cap': market_cap,
        'price_change_24h': np.random.uniform(-5, 5, n_samples),
        'volume_change_24h': np.random.uniform(-20, 20, n_samples),
        'market_cap_change_24h': np.random.uniform(-5, 5, n_samples),
    })
    
    # Add derived features
    data['price_range'] = data['high'] - data['low']
    data['price_range_pct'] = (data['price_range'] / data['price']) * 100
    data['volume_price_ratio'] = data['volume'] / data['price']
    
    # Moving averages
    data['ma_7'] = data['price'].rolling(window=7, min_periods=1).mean()
    data['ma_14'] = data['price'].rolling(window=14, min_periods=1).mean()
    data['ma_30'] = data['price'].rolling(window=30, min_periods=1).mean()
    
    # Volatility
    data['volatility_7d'] = data['price'].rolling(window=7, min_periods=1).std()
    data['volatility_14d'] = data['price'].rolling(window=14, min_periods=1).std()
    
    # Target: next price (for prediction)
    data['target_price'] = data['price'].shift(-1)
    data = data.dropna()
    
    return data


def save_synthetic_data(output_dir='data', n_samples=1000):
    """Generate and save synthetic data"""
    output_path = Path(output_dir)
    
    # Create directories
    (output_path / 'raw').mkdir(parents=True, exist_ok=True)
    (output_path / 'processed').mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print(f"Generating {n_samples} synthetic data points...")
    data = create_synthetic_crypto_data(n_samples=n_samples)
    
    # Save raw data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_file = output_path / 'raw' / f'crypto_raw_{timestamp}.csv'
    data.to_csv(raw_file, index=False)
    print(f"✓ Saved raw data: {raw_file}")
    
    # Save processed data (parquet format)
    processed_file = output_path / 'processed' / f'processed_crypto_{timestamp}.parquet'
    data.to_parquet(processed_file, index=False)
    print(f"✓ Saved processed data: {processed_file}")
    
    # Print summary
    print(f"\nData Summary:")
    print(f"  Samples: {len(data)}")
    print(f"  Features: {len(data.columns)}")
    print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"  Price range: ${data['price'].min():,.2f} - ${data['price'].max():,.2f}")
    
    return data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic crypto data')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    save_synthetic_data(output_dir=args.output, n_samples=args.samples)
    print("\n✅ Synthetic data generation complete!")


if __name__ == "__main__":
    main()
