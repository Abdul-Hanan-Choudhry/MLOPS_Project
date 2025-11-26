"""
Configuration management for the MLOps project.
Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class CoinGeckoConfig:
    """CoinGecko API configuration."""
    API_URL: str = os.getenv("COINGECKO_API_URL", "https://api.coingecko.com/api/v3")
    COIN_ID: str = os.getenv("COIN_ID", "bitcoin")
    VS_CURRENCY: str = os.getenv("VS_CURRENCY", "usd")
    DATA_FETCH_DAYS: int = int(os.getenv("DATA_FETCH_DAYS", "30"))


class MLflowConfig:
    """MLflow and DagsHub configuration."""
    DAGSHUB_USERNAME: str = os.getenv("DAGSHUB_USERNAME", "")
    DAGSHUB_REPO: str = os.getenv("DAGSHUB_REPO", "")
    TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME', '')}/{os.getenv('DAGSHUB_REPO', '')}.mlflow"
    )
    EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "crypto-price-prediction")


class StorageConfig:
    """MinIO/S3 storage configuration."""
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "mlops-data")
    USE_SSL: bool = os.getenv("MINIO_USE_SSL", "false").lower() == "true"


class DataQualityConfig:
    """Data quality validation configuration."""
    NULL_THRESHOLD: float = float(os.getenv("NULL_THRESHOLD", "0.01"))
    MIN_ROWS_REQUIRED: int = int(os.getenv("MIN_ROWS_REQUIRED", "100"))
    REQUIRED_COLUMNS: List[str] = ["timestamp", "price", "volume", "market_cap"]


class FeatureConfig:
    """Feature engineering configuration."""
    LAG_PERIODS: List[int] = [int(x) for x in os.getenv("LAG_PERIODS", "1,3,6,12,24").split(",")]
    ROLLING_WINDOWS: List[int] = [int(x) for x in os.getenv("ROLLING_WINDOWS", "6,12,24").split(",")]
    TARGET_COLUMN: str = os.getenv("TARGET_COLUMN", "target_price_1h")


class ModelConfig:
    """Model configuration."""
    MODEL_NAME: str = os.getenv("MODEL_NAME", "crypto-price-predictor")
    TARGET_COLUMN: str = os.getenv("TARGET_COLUMN", "target_price_1h")


# Create config instances
coingecko_config = CoinGeckoConfig()
mlflow_config = MLflowConfig()
storage_config = StorageConfig()
data_quality_config = DataQualityConfig()
feature_config = FeatureConfig()
model_config = ModelConfig()
