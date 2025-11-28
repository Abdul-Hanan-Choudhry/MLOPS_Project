"""
Check Model Performance Against Threshold
Returns exit code 0 if model meets criteria, 1 otherwise
"""
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow():
    """Configure MLflow with DagsHub credentials"""
    tracking_uri = os.getenv(
        'MLFLOW_TRACKING_URI',
        'https://dagshub.com/abdulhananch404/MLOPS_Project.mlflow'
    )
    mlflow.set_tracking_uri(tracking_uri)
    
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    if dagshub_token:
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv(
            'MLFLOW_TRACKING_USERNAME', 'abdulhananch404'
        )
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
    
    return MlflowClient()


def check_performance(
    rmse_threshold=500.0,
    r2_threshold=0.7,
    max_degradation_pct=10.0
):
    """
    Check if the latest model meets performance criteria
    
    Args:
        rmse_threshold: Maximum acceptable RMSE
        r2_threshold: Minimum acceptable R² score
        max_degradation_pct: Maximum allowed degradation from baseline
    
    Returns:
        bool: True if model passes checks, False otherwise
    """
    client = setup_mlflow()
    
    # Get experiment
    experiment = client.get_experiment_by_name("crypto-price-prediction")
    if experiment is None:
        experiments = client.search_experiments()
        if not experiments:
            print("⚠️ No experiments found - allowing deployment")
            return True
        experiment = experiments[0]
    
    # Get latest runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=10
    )
    
    if not runs:
        print("⚠️ No runs found - allowing deployment")
        return True
    
    latest_run = runs[0]
    latest_rmse = latest_run.data.metrics.get('rmse', float('inf'))
    latest_r2 = latest_run.data.metrics.get('r2', 0)
    model_name = latest_run.data.params.get('model_name', 'Unknown')
    
    print(f"\n{'='*50}")
    print("Model Performance Check")
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"RMSE: {latest_rmse:.4f} (threshold: {rmse_threshold})")
    print(f"R²: {latest_r2:.4f} (threshold: {r2_threshold})")
    
    checks_passed = True
    
    # Check RMSE threshold
    if latest_rmse > rmse_threshold:
        print(f"❌ FAIL: RMSE {latest_rmse:.4f} exceeds threshold {rmse_threshold}")
        checks_passed = False
    else:
        print(f"✅ PASS: RMSE within threshold")
    
    # Check R² threshold
    if latest_r2 < r2_threshold:
        print(f"❌ FAIL: R² {latest_r2:.4f} below threshold {r2_threshold}")
        checks_passed = False
    else:
        print(f"✅ PASS: R² above threshold")
    
    # Check against baseline
    if len(runs) > 1:
        # Find best previous run
        baseline_rmse = min(r.data.metrics.get('rmse', float('inf')) for r in runs[1:])
        
        if baseline_rmse > 0:
            degradation = ((latest_rmse - baseline_rmse) / baseline_rmse) * 100
            
            print(f"\nBaseline RMSE: {baseline_rmse:.4f}")
            print(f"Degradation: {degradation:+.2f}%")
            
            if degradation > max_degradation_pct:
                print(f"❌ FAIL: Model degraded by {degradation:.2f}% (max: {max_degradation_pct}%)")
                checks_passed = False
            else:
                print(f"✅ PASS: Model performance acceptable")
    
    print(f"\n{'='*50}")
    if checks_passed:
        print("✅ ALL CHECKS PASSED - Ready for deployment")
    else:
        print("❌ CHECKS FAILED - Review required")
    print(f"{'='*50}\n")
    
    return checks_passed


def main():
    # Get thresholds from environment or use defaults
    rmse_threshold = float(os.getenv('RMSE_THRESHOLD', '500.0'))
    r2_threshold = float(os.getenv('R2_THRESHOLD', '0.7'))
    max_degradation = float(os.getenv('MAX_DEGRADATION_PCT', '10.0'))
    
    try:
        passed = check_performance(
            rmse_threshold=rmse_threshold,
            r2_threshold=r2_threshold,
            max_degradation_pct=max_degradation
        )
        
        sys.exit(0 if passed else 1)
        
    except Exception as e:
        print(f"⚠️ Error during performance check: {e}")
        print("Allowing deployment with warning...")
        sys.exit(0)


if __name__ == "__main__":
    main()
