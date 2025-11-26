"""
Feature Engineering / Transformation Module
Transforms raw crypto data into features for ML model training.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import feature_config, PROCESSED_DATA_DIR, REPORTS_DIR

logger = get_logger(__name__)


class CryptoFeatureEngineer:
    """
    Transforms raw cryptocurrency data into ML-ready features.
    
    Creates:
    - Lag features (historical prices/volumes)
    - Rolling statistics (means, std deviations)
    - Time-based features (hour, day of week, etc.)
    - Technical indicators (volatility, momentum)
    - Target variable (future price)
    """
    
    def __init__(
        self,
        lag_periods: List[int] = None,
        rolling_windows: List[int] = None,
        target_horizon: int = 1
    ):
        """
        Initialize the feature engineer.
        
        Args:
            lag_periods: List of lag periods for historical features (in hours)
            rolling_windows: List of rolling window sizes for statistics
            target_horizon: Hours ahead to predict (default 1 hour)
        """
        self.lag_periods = lag_periods or feature_config.LAG_PERIODS
        self.rolling_windows = rolling_windows or feature_config.ROLLING_WINDOWS
        self.target_horizon = target_horizon
        
        logger.info(
            f"Initialized CryptoFeatureEngineer: "
            f"lag_periods={self.lag_periods}, "
            f"rolling_windows={self.rolling_windows}, "
            f"target_horizon={self.target_horizon}h"
        )
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for price and volume.
        
        Args:
            df: DataFrame with price and volume columns
            
        Returns:
            DataFrame with lag features added
        """
        logger.debug(f"Creating lag features for periods: {self.lag_periods}")
        
        df = df.copy()
        
        for lag in self.lag_periods:
            # Price lags
            df[f"price_lag_{lag}h"] = df["price"].shift(lag)
            
            # Volume lags
            df[f"volume_lag_{lag}h"] = df["volume"].shift(lag)
            
            # Price change from lag
            df[f"price_change_{lag}h"] = df["price"] - df[f"price_lag_{lag}h"]
            df[f"price_pct_change_{lag}h"] = df["price"].pct_change(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df: DataFrame with price and volume columns
            
        Returns:
            DataFrame with rolling features added
        """
        logger.debug(f"Creating rolling features for windows: {self.rolling_windows}")
        
        df = df.copy()
        
        for window in self.rolling_windows:
            # Price rolling statistics
            df[f"price_rolling_mean_{window}h"] = df["price"].rolling(window).mean()
            df[f"price_rolling_std_{window}h"] = df["price"].rolling(window).std()
            df[f"price_rolling_min_{window}h"] = df["price"].rolling(window).min()
            df[f"price_rolling_max_{window}h"] = df["price"].rolling(window).max()
            
            # Volume rolling statistics
            df[f"volume_rolling_mean_{window}h"] = df["volume"].rolling(window).mean()
            df[f"volume_rolling_std_{window}h"] = df["volume"].rolling(window).std()
            
            # Market cap rolling mean
            if "market_cap" in df.columns:
                df[f"market_cap_rolling_mean_{window}h"] = df["market_cap"].rolling(window).mean()
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with time features added
        """
        logger.debug("Creating time-based features")
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Time components
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
        
        # Binary features
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_night"] = df["hour"].isin(range(0, 6)).astype(int)
        df["is_trading_hours"] = df["hour"].isin(range(9, 17)).astype(int)
        
        # Cyclical encoding for hour (sin/cos for continuity)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        # Cyclical encoding for day of week
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        return df
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicator features.
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            DataFrame with technical indicators added
        """
        logger.debug("Creating technical indicators")
        
        df = df.copy()
        
        # Volatility (coefficient of variation over rolling window)
        for window in self.rolling_windows:
            mean_col = f"price_rolling_mean_{window}h"
            std_col = f"price_rolling_std_{window}h"
            if mean_col in df.columns and std_col in df.columns:
                df[f"volatility_{window}h"] = df[std_col] / df[mean_col]
        
        # Price momentum (only if we have enough lag)
        max_lag = max(self.lag_periods) if self.lag_periods else 24
        if max_lag >= 12:
            df["momentum_12h"] = df["price"] / df["price"].shift(12) - 1
        if max_lag >= 24:
            df["momentum_24h"] = df["price"] / df["price"].shift(24) - 1
        
        # Price position (relative to rolling min/max)
        for window in self.rolling_windows:
            min_col = f"price_rolling_min_{window}h"
            max_col = f"price_rolling_max_{window}h"
            if min_col in df.columns and max_col in df.columns:
                range_val = df[max_col] - df[min_col]
                df[f"price_position_{window}h"] = (df["price"] - df[min_col]) / range_val.replace(0, np.nan)
        
        # Volume relative strength (use largest available rolling window)
        largest_window = max(self.rolling_windows) if self.rolling_windows else 24
        vol_mean_col = f"volume_rolling_mean_{largest_window}h"
        if vol_mean_col in df.columns:
            df[f"volume_ratio_{largest_window}h"] = df["volume"] / df[vol_mean_col]
            
            # Price-volume correlation proxy
            pct_change_col = "price_pct_change_1h" if "price_pct_change_1h" in df.columns else None
            if pct_change_col:
                df["pv_momentum"] = df[pct_change_col] * df[f"volume_ratio_{largest_window}h"]
        
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            df: DataFrame with price column
            
        Returns:
            DataFrame with target column added
        """
        logger.debug(f"Creating target variable: price {self.target_horizon}h ahead")
        
        df = df.copy()
        
        # Future price (what we want to predict)
        df[f"target_price_{self.target_horizon}h"] = df["price"].shift(-self.target_horizon)
        
        # Future price change (alternative target)
        df[f"target_pct_change_{self.target_horizon}h"] = df["price"].pct_change(-self.target_horizon) * -1
        
        # Direction (binary classification alternative)
        df[f"target_direction_{self.target_horizon}h"] = (
            df[f"target_price_{self.target_horizon}h"] > df["price"]
        ).astype(int)
        
        return df
    
    def transform(
        self,
        df: pd.DataFrame,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Apply all feature transformations.
        
        Args:
            df: Raw DataFrame with timestamp, price, volume, market_cap
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            Transformed DataFrame with all features
        """
        logger.info(f"Starting feature transformation on {len(df)} rows")
        
        # Ensure sorted by timestamp
        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        
        # Apply all transformations
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_time_features(df)
        df = self.create_technical_indicators(df)
        df = self.create_target(df)
        
        rows_before = len(df)
        
        if drop_na:
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            logger.info(f"Dropped {rows_dropped} rows with NaN values")
        
        logger.info(
            f"Transformation complete: {len(df)} rows, {len(df.columns)} columns"
        )
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding metadata and targets).
        
        Args:
            df: Transformed DataFrame
            
        Returns:
            List of feature column names
        """
        exclude_patterns = [
            "timestamp", "extraction_time", "coin_id", "vs_currency",
            "target_", "price", "volume", "market_cap"
        ]
        
        # Exclude exact matches and pattern matches
        feature_cols = []
        for col in df.columns:
            is_excluded = False
            for pattern in exclude_patterns:
                if pattern in col and not any(
                    f"_{pattern}" in col or f"{pattern}_" in col 
                    for pattern in ["lag", "rolling", "pct"]
                ):
                    if col in ["price", "volume", "market_cap"]:
                        is_excluded = True
                        break
                    if col.startswith("target_"):
                        is_excluded = True
                        break
                    if col in ["timestamp", "extraction_time", "coin_id", "vs_currency"]:
                        is_excluded = True
                        break
            
            if not is_excluded:
                feature_cols.append(col)
        
        # Filter out metadata columns
        feature_cols = [c for c in feature_cols 
                       if c not in ["timestamp", "extraction_time", "coin_id", "vs_currency"]
                       and not c.startswith("target_")]
        
        return feature_cols
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        output_dir: Path = None,
        file_format: str = "parquet"
    ) -> Path:
        """
        Save processed data to disk.
        
        Args:
            df: Processed DataFrame
            output_dir: Output directory
            file_format: File format ('parquet' or 'csv')
            
        Returns:
            Path to saved file
        """
        output_dir = output_dir or PROCESSED_DATA_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_crypto_{timestamp}.{file_format}"
        filepath = output_dir / filename
        
        if file_format == "parquet":
            df.to_parquet(filepath, index=False)
        elif file_format == "csv":
            df.to_csv(filepath, index=False)
        
        logger.info(f"Saved processed data to {filepath}")
        
        return filepath


def generate_data_profile_report(
    df: pd.DataFrame,
    output_dir: Path = None,
    title: str = "Crypto Features Data Profile"
) -> Path:
    """
    Generate a data profiling report using ydata-profiling.
    Falls back to a simple HTML report if ydata-profiling fails.
    
    Args:
        df: DataFrame to profile
        output_dir: Output directory for report
        title: Report title
        
    Returns:
        Path to generated report
    """
    output_dir = output_dir or REPORTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"data_profile_{timestamp}.html"
    filepath = output_dir / filename
    
    # Try ydata-profiling first
    try:
        from ydata_profiling import ProfileReport
        
        logger.info(f"Generating data profile report with ydata-profiling: {filepath}")
        
        # Use minimal mode to avoid compatibility issues
        profile = ProfileReport(
            df,
            title=title,
            minimal=True,  # Use minimal mode for stability
            explorative=False,
            correlations=None,  # Disable correlations to avoid ndim error
            missing_diagrams=None
        )
        
        profile.to_file(filepath)
        logger.info(f"Data profile report saved to {filepath}")
        return filepath
        
    except ImportError:
        logger.warning("ydata-profiling not installed, generating simple report")
    except Exception as e:
        logger.warning(f"ydata-profiling failed ({e}), generating simple HTML report")
    
    # Fallback: Generate simple HTML report
    try:
        logger.info(f"Generating simple HTML data profile report: {filepath}")
        html_content = generate_simple_profile_html(df, title)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Simple data profile report saved to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate profile report: {e}")
        return None


def generate_simple_profile_html(df: pd.DataFrame, title: str) -> str:
    """
    Generate a simple HTML data profile report.
    
    Args:
        df: DataFrame to profile
        title: Report title
        
    Returns:
        HTML string
    """
    # Basic statistics
    stats = df.describe().round(4)
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    dtypes = df.dtypes
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .summary-box {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2e7d32; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .warning {{ color: #f57c00; }}
        .good {{ color: #388e3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {title}</h1>
        <p>Generated on: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
        
        <div class="summary-box">
            <div class="metric">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(df.columns)}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            <div class="metric">
                <div class="metric-value">{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{null_counts.sum():,}</div>
                <div class="metric-label">Missing Values</div>
            </div>
        </div>
        
        <h2>📋 Column Information</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Column Name</th>
                <th>Data Type</th>
                <th>Non-Null Count</th>
                <th>Null Count</th>
                <th>Null %</th>
            </tr>
    """
    
    for i, col in enumerate(df.columns, 1):
        null_class = "warning" if null_pct[col] > 5 else "good"
        html += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{col}</strong></td>
                <td>{dtypes[col]}</td>
                <td>{len(df) - null_counts[col]:,}</td>
                <td>{null_counts[col]:,}</td>
                <td class="{null_class}">{null_pct[col]}%</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>📈 Numerical Statistics</h2>
        <table>
            <tr>
                <th>Statistic</th>
    """
    
    # Add column headers for numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns[:15]  # Limit to first 15
    for col in numeric_cols:
        html += f"<th>{col[:20]}</th>"
    
    html += "</tr>"
    
    # Add statistics rows
    for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        html += f"<tr><td><strong>{stat}</strong></td>"
        for col in numeric_cols:
            if col in stats.columns:
                val = stats.loc[stat, col]
                if abs(val) > 1000:
                    html += f"<td>{val:,.2f}</td>"
                else:
                    html += f"<td>{val:.4f}</td>"
            else:
                html += "<td>-</td>"
        html += "</tr>"
    
    html += """
        </table>
        
        <h2>🎯 Target Variable Analysis</h2>
    """
    
    # Add target variable info if exists
    if 'target_price_1h' in df.columns:
        target = df['target_price_1h']
        html += f"""
        <div class="summary-box">
            <p><strong>Target Column:</strong> target_price_1h</p>
            <p><strong>Mean:</strong> {target.mean():.4f}</p>
            <p><strong>Std:</strong> {target.std():.4f}</p>
            <p><strong>Min:</strong> {target.min():.4f}</p>
            <p><strong>Max:</strong> {target.max():.4f}</p>
        </div>
        """
    
    html += """
        <h2>📦 Feature Categories</h2>
        <table>
            <tr><th>Category</th><th>Count</th><th>Features</th></tr>
    """
    
    # Categorize features
    categories = {
        'Lag Features': [c for c in df.columns if 'lag' in c.lower()],
        'Rolling Features': [c for c in df.columns if 'rolling' in c.lower()],
        'Time Features': [c for c in df.columns if any(x in c.lower() for x in ['hour', 'day', 'week', 'month', 'is_'])],
        'Technical Indicators': [c for c in df.columns if any(x in c.lower() for x in ['rsi', 'macd', 'bb_', 'ema', 'sma', 'momentum', 'volatility'])],
        'Base Features': [c for c in df.columns if c in ['timestamp', 'price', 'market_cap', 'total_volume']],
        'Other': []
    }
    
    # Find uncategorized
    categorized = set()
    for cat_cols in categories.values():
        categorized.update(cat_cols)
    categories['Other'] = [c for c in df.columns if c not in categorized]
    
    for cat_name, cat_cols in categories.items():
        if cat_cols:
            cols_display = ', '.join(cat_cols[:5])
            if len(cat_cols) > 5:
                cols_display += f'... (+{len(cat_cols)-5} more)'
            html += f"<tr><td>{cat_name}</td><td>{len(cat_cols)}</td><td>{cols_display}</td></tr>"
    
    html += """
        </table>
        
        <footer style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
            <p>MLOps Project - Crypto Price Prediction Pipeline</p>
            <p>Report generated by automated ETL pipeline</p>
        </footer>
    </div>
</body>
</html>
    """
    
    return html


def transform_data(
    df: pd.DataFrame,
    save: bool = True,
    generate_report: bool = True,
    output_dir: Path = None,
    **kwargs
) -> tuple[pd.DataFrame, Optional[Path], Optional[Path]]:
    """
    Main transformation function for use in pipeline.
    
    Args:
        df: Raw DataFrame to transform
        save: Whether to save processed data
        generate_report: Whether to generate profiling report
        output_dir: Output directory
        **kwargs: Additional arguments for feature engineer
        
    Returns:
        Tuple of (transformed_df, data_filepath, report_filepath)
    """
    engineer = CryptoFeatureEngineer(**kwargs)
    df_transformed = engineer.transform(df)
    
    data_filepath = None
    report_filepath = None
    
    if save:
        data_filepath = engineer.save_processed_data(df_transformed, output_dir)
    
    if generate_report:
        report_filepath = generate_data_profile_report(df_transformed)
    
    return df_transformed, data_filepath, report_filepath


if __name__ == "__main__":
    # Test the feature engineer
    logger.info("Running feature engineering test...")
    
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=200, freq="H")
    test_df = pd.DataFrame({
        "timestamp": dates,
        "price": 45000 + np.cumsum(np.random.randn(200) * 100),
        "volume": np.random.uniform(1e9, 2e9, 200),
        "market_cap": np.random.uniform(8e11, 9e11, 200)
    })
    
    print(f"\n{'='*50}")
    print("Feature Engineering Test")
    print(f"{'='*50}")
    print(f"Input shape: {test_df.shape}")
    
    engineer = CryptoFeatureEngineer()
    df_transformed = engineer.transform(test_df)
    
    print(f"Output shape: {df_transformed.shape}")
    print(f"\nFeature columns ({len(engineer.get_feature_columns(df_transformed))}):")
    
    feature_cols = engineer.get_feature_columns(df_transformed)
    for i, col in enumerate(feature_cols[:20], 1):
        print(f"  {i}. {col}")
    if len(feature_cols) > 20:
        print(f"  ... and {len(feature_cols) - 20} more")
    
    print(f"\nTarget columns:")
    target_cols = [c for c in df_transformed.columns if c.startswith("target_")]
    for col in target_cols:
        print(f"  - {col}")
    
    print(f"\nSample transformed data:")
    print(df_transformed[["timestamp", "price", "price_lag_1h", "volatility_24h", "target_price_1h"]].head())
