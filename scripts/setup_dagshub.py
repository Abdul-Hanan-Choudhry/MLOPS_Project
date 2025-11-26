"""
DagsHub Setup Helper Script for MLOps Project

This script helps you configure DagsHub integration for MLflow tracking.
Run this script after creating your DagsHub account and repository.

Usage:
    python scripts/setup_dagshub.py
"""

import os
import sys
from pathlib import Path

def print_banner():
    print("\n" + "=" * 60)
    print("   DAGSHUB SETUP FOR MLOPS PROJECT")
    print("=" * 60 + "\n")

def print_step(step_num, title):
    print(f"\n📌 STEP {step_num}: {title}")
    print("-" * 50)

def main():
    print_banner()
    
    print("""
This script will help you set up DagsHub for MLflow tracking.
The Pandas Profiling report will be logged as an artifact to DagsHub.

Prerequisites:
✓ A DagsHub account (free at https://dagshub.com)
✓ A repository created on DagsHub
    """)
    
    print_step(1, "Create DagsHub Account")
    print("""
1. Go to https://dagshub.com
2. Sign up with GitHub, Google, or email
3. Complete your profile setup
    """)
    
    print_step(2, "Create a New Repository")
    print("""
1. Click "New Repository" on DagsHub
2. Name it: MLOPS_Project
3. Description: Real-Time Crypto Price Prediction MLOps Pipeline
4. Keep it Public or Private as you prefer
5. DO NOT initialize with README (we have existing code)
6. Click "Create Repository"
    """)
    
    print_step(3, "Get Your DagsHub Token")
    print("""
1. Click your profile picture → Settings
2. Go to "Access Tokens" 
3. Click "Generate New Token"
4. Name it: "mlops-project-token"
5. Copy the token (you won't see it again!)
    """)
    
    print_step(4, "Update Your .env File")
    print("""
Open .env file in the project root and update:

    DAGSHUB_USERNAME=your_actual_username
    DAGSHUB_TOKEN=your_actual_token
    DAGSHUB_REPO=MLOPS_Project
    MLFLOW_TRACKING_URI=https://dagshub.com/your_actual_username/MLOPS_Project.mlflow
    
Example (if your username is "john_doe"):

    DAGSHUB_USERNAME=john_doe
    DAGSHUB_TOKEN=abc123xyz789...
    DAGSHUB_REPO=MLOPS_Project
    MLFLOW_TRACKING_URI=https://dagshub.com/john_doe/MLOPS_Project.mlflow
    """)
    
    print_step(5, "Restart Docker Services")
    print("""
After updating .env, restart your Docker services:

    cd docker
    docker-compose down
    docker-compose up -d
    
Wait for services to be healthy:

    docker-compose ps
    """)
    
    print_step(6, "Trigger the DAG")
    print("""
1. Open Airflow UI: http://localhost:8085
2. Login: admin / admin
3. Find "crypto_etl_training_pipeline" DAG
4. Click the play button to trigger a run
5. Monitor the tasks, especially "log_to_mlflow"
    """)
    
    print_step(7, "View Results on DagsHub")
    print("""
1. Go to your DagsHub repository
2. Click the "Experiments" tab
3. You should see your MLflow runs with:
   - Parameters (coin_id, days, etc.)
   - Metrics (processed_rows, feature_count, price_mean, etc.)
   - Artifacts:
     • data_quality_reports/data_profile_XXXXXX.html  ← Pandas Profiling!
     • validation/validation_report.json
     • data_samples/data_sample.csv
     • documentation/feature_list.txt
    """)
    
    print("\n" + "=" * 60)
    print("   QUICK VERIFICATION TEST")
    print("=" * 60)
    
    print("""
After setup, run this test to verify DagsHub connection:

    python -c "
import mlflow
import os

os.environ['MLFLOW_TRACKING_USERNAME'] = 'your_username'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'your_token'

mlflow.set_tracking_uri('https://dagshub.com/your_username/MLOPS_Project.mlflow')
mlflow.set_experiment('test-connection')

with mlflow.start_run(run_name='connection-test'):
    mlflow.log_param('test', 'success')
    print('✅ DagsHub connection successful!')
"
    """)
    
    print("\n" + "=" * 60)
    print("   TROUBLESHOOTING")
    print("=" * 60)
    
    print("""
❌ "401 Unauthorized" error:
   → Check your DAGSHUB_USERNAME and DAGSHUB_TOKEN are correct
   → Ensure the token has read/write permissions

❌ "Experiment not found" error:
   → The experiment will be created automatically
   → Make sure MLFLOW_EXPERIMENT_NAME is set

❌ "Connection refused" error:
   → Check MLFLOW_TRACKING_URI format
   → Should be: https://dagshub.com/USERNAME/REPO.mlflow

❌ No profile report logged:
   → Check if ydata-profiling is installed in Docker
   → Run: docker exec docker-airflow-scheduler-1 pip list | grep ydata
    """)
    
    print("\n✅ Setup guide complete! Follow the steps above.\n")


if __name__ == "__main__":
    main()
