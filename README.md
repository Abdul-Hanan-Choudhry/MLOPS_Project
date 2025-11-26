# MLOps Project: Real-Time Crypto Price Prediction System

A production-grade MLOps pipeline for predicting cryptocurrency prices using data from the CoinGecko API. Built with Apache Airflow for orchestration, MLflow for experiment tracking, and DVC for data versioning.

## 🎯 Project Overview

**Goal:** Predict short-term cryptocurrency price movements (1-hour ahead) using real-time market data.

**Domain:** Financial - Cryptocurrency Markets  
**API:** CoinGecko (Free tier)  
**Prediction Task:** Next-hour price prediction for Bitcoin (or other cryptocurrencies)

## 📁 Project Structure

```
MLOPS_Project/
├── dags/                           # Airflow DAG definitions
│   └── crypto_etl_pipeline.py      # Main ETL and training DAG
├── src/
│   ├── data/
│   │   ├── extract.py              # CoinGecko API data extraction
│   │   ├── validate.py             # Data quality validation
│   │   └── transform.py            # Feature engineering
│   ├── models/
│   │   └── train.py                # Model training (Phase 2)
│   ├── api/
│   │   └── app.py                  # FastAPI prediction server (Phase 4)
│   └── utils/
│       ├── config.py               # Configuration management
│       └── logger.py               # Logging utilities
├── tests/                          # Unit tests
├── docker/
│   ├── docker-compose.yml          # Airflow + MinIO setup
│   └── Dockerfile.airflow          # Custom Airflow image
├── monitoring/                     # Prometheus & Grafana configs
├── data/
│   ├── raw/                        # Raw extracted data
│   ├── processed/                  # Feature-engineered data
│   └── reports/                    # Data profiling reports
├── .dvc/                           # DVC configuration
├── dvc.yaml                        # DVC pipeline definition
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Git

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd MLOPS_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### 2. Initialize DVC

```bash
# Initialize DVC (already done, but for new setups)
dvc init

# Configure remote storage (MinIO)
dvc remote add -d minio s3://dvc-storage
dvc remote modify minio endpointurl http://localhost:9000
```

### 3. Start Services with Docker

```bash
# Navigate to docker directory
cd docker

# Start all services (Airflow, MinIO, PostgreSQL)
docker-compose up -d

# Wait for services to be ready (2-3 minutes)
docker-compose logs -f airflow-webserver
```

### 4. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | airflow / airflow |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

### 5. Run the Pipeline

#### Option A: Via Airflow UI
1. Open http://localhost:8080
2. Enable the `crypto_etl_training_pipeline` DAG
3. Trigger manually or wait for scheduled run

#### Option B: Run Locally (for testing)

```bash
# Test extraction
python -m src.data.extract

# Test validation
python -m src.data.validate

# Test transformation
python -m src.data.transform
```

## 📊 Phase 1: Data Ingestion Pipeline

### Components

#### 1. Data Extraction (`src/data/extract.py`)
- Connects to CoinGecko API
- Fetches historical market data (prices, volume, market cap)
- Saves raw data with extraction timestamp
- Supports configurable coin ID and time range

```python
from src.data.extract import extract_data

# Extract 30 days of Bitcoin data
df, filepath = extract_data(coin_id="bitcoin", days=30)
```

#### 2. Data Validation (`src/data/validate.py`)
- **Schema validation**: Ensures required columns exist
- **Null check**: Fails if >1% null values in key columns
- **Row count**: Minimum 100 rows required
- **Value ranges**: Validates price, volume are positive
- **Duplicates**: Checks for duplicate timestamps

```python
from src.data.validate import validate_data

# Validate with mandatory quality gate
report = validate_data(df, fail_on_error=True)  # Raises on failure
```

#### 3. Feature Engineering (`src/data/transform.py`)

Creates features for ML model:

| Feature Type | Examples |
|--------------|----------|
| **Lag Features** | price_lag_1h, price_lag_24h, volume_lag_6h |
| **Rolling Stats** | price_rolling_mean_24h, price_rolling_std_12h |
| **Time Features** | hour, day_of_week, is_weekend, hour_sin, hour_cos |
| **Technical** | volatility_24h, momentum_12h, price_position |
| **Target** | target_price_1h (next hour price) |

```python
from src.data.transform import transform_data

# Transform and generate profile report
df_features, data_path, report_path = transform_data(df)
```

#### 4. Data Profiling
- Uses `ydata-profiling` for detailed data analysis
- Generates HTML reports logged to MLflow
- Includes statistics, distributions, correlations

### Airflow DAG Structure

```
extract_data → validate_data → transform_features
                                      ↓
                         ┌────────────┴────────────┐
                         ↓                         ↓
              generate_profile_report      version_with_dvc
                         └────────────┬────────────┘
                                      ↓
                              log_to_mlflow
                                      ↓
                             trigger_training
                                      ↓
                            pipeline_complete
```

### Data Quality Gate

The validation step is a **mandatory gate**. If any critical check fails:
- The DAG stops immediately
- No downstream tasks execute
- Error is logged with details

Critical checks (fail pipeline):
- Missing required columns
- >1% null values
- Fewer than 100 rows
- Negative prices or volumes

## 🗄️ Data Versioning with DVC

### Setup

```bash
# Track processed data
dvc add data/processed/

# Commit DVC metadata
git add data/processed.dvc .gitignore
git commit -m "Add processed data"

# Push data to remote storage
dvc push
```

### Usage

```bash
# Pull data
dvc pull

# Reproduce pipeline
dvc repro

# Check status
dvc status
```

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# CoinGecko API
COIN_ID=bitcoin
VS_CURRENCY=usd
DATA_FETCH_DAYS=30

# MLflow/DagsHub
DAGSHUB_USERNAME=your_username
DAGSHUB_REPO=your_repo
MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow

# MinIO (S3-compatible storage)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Data Quality
NULL_THRESHOLD=0.01
MIN_ROWS_REQUIRED=100
```

### Feature Configuration

```python
# In src/utils/config.py
LAG_PERIODS = [1, 3, 6, 12, 24]  # Hours
ROLLING_WINDOWS = [6, 12, 24]    # Hours
TARGET_HORIZON = 1               # Predict 1 hour ahead
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_extract.py -v
```

## 📈 Next Phases

### Phase 2: Model Training & MLflow
- Implement `src/models/train.py`
- MLflow experiment tracking
- Model registry on DagsHub

### Phase 3: CI/CD
- GitHub Actions workflows
- CML for automated model comparison
- Branch protection rules

### Phase 4: Deployment & Monitoring
- FastAPI prediction server
- Prometheus metrics
- Grafana dashboards

## 🤝 Contributing

1. Create a feature branch from `dev`
2. Make changes and add tests
3. Submit PR for review
4. Merge to `dev` → `test` → `master`

## 📝 License

This project is for educational purposes as part of the MLOps course.

---

**Deadline:** November 30, 2025
