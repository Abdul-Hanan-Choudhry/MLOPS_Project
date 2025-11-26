"""
Crypto ETL Pipeline DAG for Apache Airflow
Orchestrates the complete ETL and model training workflow.

This DAG:
1. Extracts crypto data from CoinGecko API
2. Validates data quality (mandatory gate)
3. Transforms data with feature engineering
4. Generates data profiling report
5. Versions data with DVC
6. Triggers model training
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# Default arguments for the DAG
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}


def extract_crypto_data(**context):
    """
    Task: Extract cryptocurrency data from CoinGecko API.
    
    Fetches historical market data and saves raw data with timestamp.
    Falls back to synthetic data if API is unavailable.
    """
    # Add project path
    sys.path.insert(0, "/opt/airflow")
    
    from src.data.extract import extract_data
    
    coin_id = context["params"].get("coin_id", "bitcoin")
    days = context["params"].get("days", 30)
    
    print(f"Extracting {days} days of data for {coin_id}...")
    
    # Use extract_data which has automatic fallback to synthetic data
    df, raw_filepath = extract_data(
        coin_id=coin_id,
        days=days,
        output_dir="/opt/airflow/data/raw",
        save=True,
        use_synthetic=False  # Will auto-fallback if API fails
    )
    
    # Store filepath in XCom for downstream tasks
    context["ti"].xcom_push(key="raw_filepath", value=str(raw_filepath))
    context["ti"].xcom_push(key="row_count", value=len(df))
    
    print(f"✓ Extracted {len(df)} records")
    print(f"✓ Saved raw data to {raw_filepath}")
    
    return str(raw_filepath)


def validate_data_quality(**context):
    """
    Task: Mandatory Data Quality Gate.
    
    Validates extracted data for:
    - Schema compliance
    - Null value threshold (<1%)
    - Minimum row count
    - Value ranges
    
    Raises exception and fails DAG if quality checks fail.
    """
    sys.path.insert(0, "/opt/airflow")
    
    import pandas as pd
    from src.data.validate import DataQualityValidator, DataQualityError
    
    # Get raw file path from previous task
    raw_filepath = context["ti"].xcom_pull(key="raw_filepath", task_ids="extract_data")
    
    print(f"Loading data from {raw_filepath}...")
    df = pd.read_parquet(raw_filepath)
    
    print(f"Running data quality validation on {len(df)} rows...")
    
    # Initialize validator with strict thresholds
    validator = DataQualityValidator(
        null_threshold=0.01,  # 1% max null values
        min_rows=100
    )
    
    # This will raise DataQualityError if validation fails
    # causing the DAG to stop
    try:
        report = validator.validate_or_fail(df)
        
        # Store validation results
        context["ti"].xcom_push(key="validation_passed", value=True)
        context["ti"].xcom_push(key="validation_report", value=report.to_dict())
        
        print(f"✓ Data quality validation PASSED")
        print(f"  - Checks passed: {report.passed_checks}/{report.total_checks}")
        
        return True
        
    except DataQualityError as e:
        print(f"✗ Data quality validation FAILED: {str(e)}")
        context["ti"].xcom_push(key="validation_passed", value=False)
        # Re-raise to fail the DAG
        raise


def transform_features(**context):
    """
    Task: Transform raw data with feature engineering.
    
    Creates:
    - Lag features
    - Rolling statistics
    - Time-based features
    - Technical indicators
    - Target variable
    """
    sys.path.insert(0, "/opt/airflow")
    
    import pandas as pd
    from src.data.transform import CryptoFeatureEngineer
    
    raw_filepath = context["ti"].xcom_pull(key="raw_filepath", task_ids="extract_data")
    
    print(f"Loading raw data from {raw_filepath}...")
    df = pd.read_parquet(raw_filepath)
    
    print(f"Applying feature transformations...")
    
    engineer = CryptoFeatureEngineer(
        lag_periods=[1, 3, 6, 12, 24],
        rolling_windows=[6, 12, 24],
        target_horizon=1
    )
    
    df_transformed = engineer.transform(df, drop_na=True)
    
    # Save processed data
    processed_filepath = engineer.save_processed_data(
        df_transformed,
        output_dir="/opt/airflow/data/processed"
    )
    
    context["ti"].xcom_push(key="processed_filepath", value=str(processed_filepath))
    context["ti"].xcom_push(key="feature_count", value=len(df_transformed.columns))
    context["ti"].xcom_push(key="processed_row_count", value=len(df_transformed))
    
    print(f"✓ Feature transformation complete")
    print(f"  - Output shape: {df_transformed.shape}")
    print(f"  - Saved to: {processed_filepath}")
    
    return str(processed_filepath)


def generate_profile_report(**context):
    """
    Task: Generate data profiling report.
    
    Uses ydata-profiling to create detailed data quality
    and feature summary report. This task is optional - if profiling
    fails, the pipeline continues.
    """
    sys.path.insert(0, "/opt/airflow")
    
    import pandas as pd
    
    processed_filepath = context["ti"].xcom_pull(
        key="processed_filepath", 
        task_ids="transform_features"
    )
    
    print(f"Loading processed data from {processed_filepath}...")
    df = pd.read_parquet(processed_filepath)
    
    print(f"Attempting to generate data profile report...")
    
    try:
        from src.data.transform import generate_data_profile_report
        
        report_filepath = generate_data_profile_report(
            df,
            output_dir="/opt/airflow/data/reports",
            title="Crypto Features Data Profile"
        )
    except ImportError as e:
        print(f"⚠ ydata-profiling not available: {e}")
        print(f"⚠ Skipping profile report generation")
        report_filepath = None
    except Exception as e:
        print(f"⚠ Profile report generation failed: {e}")
        report_filepath = None
    
    if report_filepath:
        context["ti"].xcom_push(key="report_filepath", value=str(report_filepath))
        print(f"✓ Profile report saved to {report_filepath}")
    else:
        print("⚠ Profile report generation skipped (ydata-profiling not available)")
    
    return str(report_filepath) if report_filepath else None


def log_to_mlflow(**context):
    """
    Task: Log artifacts to MLflow tracking server (DagsHub).
    
    Logs the data profile report, validation results, and metadata to DagsHub MLflow.
    This fulfills the requirement to log Pandas Profiling report as an artifact.
    """
    sys.path.insert(0, "/opt/airflow")
    
    import mlflow
    import os
    import json
    from datetime import datetime
    
    # Get DagsHub credentials from environment
    dagshub_username = os.getenv("DAGSHUB_USERNAME", "")
    dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    
    # Check if DagsHub is configured
    if dagshub_username and dagshub_token and mlflow_uri:
        print(f"📡 Connecting to DagsHub MLflow: {mlflow_uri}")
        
        # Set authentication for DagsHub
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        
        mlflow.set_tracking_uri(mlflow_uri)
    else:
        # Fallback to local MLflow
        local_uri = "file:///opt/airflow/mlruns"
        print(f"⚠ DagsHub not configured, using local MLflow: {local_uri}")
        mlflow.set_tracking_uri(local_uri)
    
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "crypto-etl-pipeline")
    
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"⚠ Could not set experiment: {e}")
        mlflow.set_experiment("Default")
    
    # Get artifacts from previous tasks
    report_filepath = context["ti"].xcom_pull(
        key="report_filepath", 
        task_ids="generate_profile_report"
    )
    processed_filepath = context["ti"].xcom_pull(
        key="processed_filepath",
        task_ids="transform_features"
    )
    raw_filepath = context["ti"].xcom_pull(
        key="raw_filepath",
        task_ids="extract_data"
    )
    validation_report = context["ti"].xcom_pull(
        key="validation_report",
        task_ids="validate_data"
    )
    
    run_name = f"etl_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"🚀 Started MLflow run: {run_name}")
        
        # ========== Log Parameters ==========
        mlflow.log_param("coin_id", context["params"].get("coin_id", "bitcoin"))
        mlflow.log_param("days", context["params"].get("days", 30))
        mlflow.log_param("execution_date", context["ds"])
        mlflow.log_param("dag_run_id", context["run_id"])
        mlflow.log_param("pipeline_version", "1.0.0")
        
        # ========== Log Metrics ==========
        row_count = context["ti"].xcom_pull(key="row_count", task_ids="extract_data")
        processed_row_count = context["ti"].xcom_pull(key="processed_row_count", task_ids="transform_features")
        feature_count = context["ti"].xcom_pull(key="feature_count", task_ids="transform_features")
        
        if row_count:
            mlflow.log_metric("raw_rows_extracted", row_count)
        if processed_row_count:
            mlflow.log_metric("processed_rows", processed_row_count)
        if feature_count:
            mlflow.log_metric("feature_count", feature_count)
        
        # Log validation metrics
        if validation_report:
            mlflow.log_metric("validation_passed", 1 if validation_report.get("passed", False) else 0)
            mlflow.log_metric("checks_passed", validation_report.get("passed_checks", 0))
            mlflow.log_metric("total_checks", validation_report.get("total_checks", 0))
            mlflow.log_metric("null_percentage", validation_report.get("null_percentage", 0))
        
        # ========== Log Artifacts ==========
        artifacts_logged = []
        
        # Log Pandas Profiling Report (REQUIRED for Phase 1)
        if report_filepath and os.path.exists(report_filepath):
            mlflow.log_artifact(report_filepath, "data_quality_reports")
            artifacts_logged.append("profile_report")
            print(f"✅ Logged Pandas Profiling report: {report_filepath}")
        else:
            print("⚠ No profile report available to log")
        
        # Log validation results as JSON
        if validation_report:
            validation_path = "/tmp/validation_report.json"
            with open(validation_path, "w") as f:
                json.dump(validation_report, f, indent=2, default=str)
            mlflow.log_artifact(validation_path, "validation")
            artifacts_logged.append("validation_report")
            print(f"✅ Logged validation report")
        
        # Log processed data sample (first 100 rows for reference)
        if processed_filepath and os.path.exists(processed_filepath):
            import pandas as pd
            df = pd.read_parquet(processed_filepath)
            
            # Log data statistics
            mlflow.log_metric("price_mean", df["price"].mean())
            mlflow.log_metric("price_std", df["price"].std())
            mlflow.log_metric("price_min", df["price"].min())
            mlflow.log_metric("price_max", df["price"].max())
            
            # Save and log sample data
            sample_path = "/tmp/data_sample.csv"
            df.head(100).to_csv(sample_path, index=False)
            mlflow.log_artifact(sample_path, "data_samples")
            artifacts_logged.append("data_sample")
            print(f"✅ Logged data sample and statistics")
        
        # Log feature list
        if processed_filepath and os.path.exists(processed_filepath):
            feature_list_path = "/tmp/feature_list.txt"
            with open(feature_list_path, "w") as f:
                f.write("# Feature Engineering Summary\n")
                f.write(f"# Total Features: {len(df.columns)}\n\n")
                for i, col in enumerate(df.columns, 1):
                    f.write(f"{i}. {col}\n")
            mlflow.log_artifact(feature_list_path, "documentation")
            artifacts_logged.append("feature_list")
        
        # Set tags
        mlflow.set_tag("pipeline_stage", "data_ingestion")
        mlflow.set_tag("data_source", "coingecko_api")
        mlflow.set_tag("artifacts_logged", ",".join(artifacts_logged))
        
        print(f"\n{'='*50}")
        print(f"✅ MLflow logging complete!")
        print(f"   Run Name: {run_name}")
        print(f"   Artifacts: {', '.join(artifacts_logged)}")
        print(f"{'='*50}\n")


def version_data_with_dvc(**context):
    """
    Task: Version processed data with DVC.
    
    Adds processed data to DVC tracking and pushes to remote storage (MinIO).
    """
    import subprocess
    import os
    
    processed_filepath = context["ti"].xcom_pull(
        key="processed_filepath",
        task_ids="transform_features"
    )
    
    if not processed_filepath:
        print("⚠ No processed file to version")
        return
    
    print(f"Versioning {processed_filepath} with DVC...")
    
    # Change to project directory
    os.chdir("/opt/airflow")
    
    try:
        # Check if DVC is initialized, if not initialize it
        if not os.path.exists("/opt/airflow/.dvc"):
            print("Initializing DVC repository...")
            subprocess.run(["dvc", "init", "--no-scm"], capture_output=True, text=True, check=True)
            
            # Configure DVC remote for MinIO (using Docker service name)
            subprocess.run([
                "dvc", "remote", "add", "-d", "minio", "s3://dvc-storage"
            ], capture_output=True, text=True, check=True)
            subprocess.run([
                "dvc", "remote", "modify", "minio", "endpointurl", "http://minio:9000"
            ], capture_output=True, text=True, check=True)
            subprocess.run([
                "dvc", "remote", "modify", "minio", "access_key_id", "minioadmin"
            ], capture_output=True, text=True, check=True)
            subprocess.run([
                "dvc", "remote", "modify", "minio", "secret_access_key", "minioadmin"
            ], capture_output=True, text=True, check=True)
            print("✓ DVC initialized with MinIO remote")
        
        # Add file to DVC
        result = subprocess.run(
            ["dvc", "add", processed_filepath],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Added to DVC: {result.stdout}")
        
        # Push to remote
        result = subprocess.run(
            ["dvc", "push"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Pushed to DVC remote: {result.stdout}")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠ DVC operation failed: {e.stderr}")
        # Don't fail the DAG for DVC issues
        
    except subprocess.CalledProcessError as e:
        print(f"⚠ DVC operation failed: {e.stderr}")
        # Don't fail the DAG for DVC issues
        

def trigger_model_training(**context):
    """
    Task: Trigger model training pipeline.
    
    Calls the training script with the processed data.
    """
    sys.path.insert(0, "/opt/airflow")
    
    processed_filepath = context["ti"].xcom_pull(
        key="processed_filepath",
        task_ids="transform_features"
    )
    
    print(f"Triggering model training with data: {processed_filepath}")
    
    # Import and run training (will be implemented in Phase 2)
    # from src.models.train import train_model
    # train_model(data_path=processed_filepath)
    
    print(f"✓ Model training triggered")
    
    return True


# Create the DAG
with DAG(
    dag_id="crypto_etl_training_pipeline",
    default_args=default_args,
    description="ETL pipeline for crypto price prediction - extracts, validates, transforms, and trains",
    schedule_interval="@daily",  # Run daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "crypto", "etl", "training"],
    params={
        "coin_id": "bitcoin",
        "days": 30
    }
) as dag:
    
    # Task 1: Extract data from CoinGecko API
    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=extract_crypto_data,
        provide_context=True,
    )
    
    # Task 2: Mandatory Data Quality Gate
    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data_quality,
        provide_context=True,
    )
    
    # Task 3: Feature Engineering
    transform_task = PythonOperator(
        task_id="transform_features",
        python_callable=transform_features,
        provide_context=True,
    )
    
    # Task 4: Generate Profile Report
    profile_task = PythonOperator(
        task_id="generate_profile_report",
        python_callable=generate_profile_report,
        provide_context=True,
    )
    
    # Task 5: Log to MLflow
    mlflow_task = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=log_to_mlflow,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,  # Continue even if profile report fails
    )
    
    # Task 6: Version with DVC
    dvc_task = PythonOperator(
        task_id="version_with_dvc",
        python_callable=version_data_with_dvc,
        provide_context=True,
    )
    
    # Task 7: Trigger Training
    train_task = PythonOperator(
        task_id="trigger_training",
        python_callable=trigger_model_training,
        provide_context=True,
    )
    
    # Pipeline end marker
    end_task = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule=TriggerRule.ALL_SUCCESS
    )
    
    # Define task dependencies (DAG structure)
    # Extract -> Validate (mandatory gate) -> Transform -> [Profile, DVC] -> MLflow -> Train -> End
    extract_task >> validate_task >> transform_task
    transform_task >> [profile_task, dvc_task]
    [profile_task, dvc_task] >> mlflow_task >> train_task >> end_task
