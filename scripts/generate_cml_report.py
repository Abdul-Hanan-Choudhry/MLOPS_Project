"""
Generate CML Report for Model Comparison
Creates markdown report comparing new model performance against baseline
"""
import os
import sys
import json
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
from pathlib import Path


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


def get_recent_runs(client, experiment_name="crypto-price-prediction", limit=20):
    """Get recent training runs"""
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiments = client.search_experiments()
        if not experiments:
            return []
        experiment = experiments[0]
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=limit
    )
    
    return runs


def get_baseline_metrics(runs):
    """Get baseline metrics from previous best run"""
    if len(runs) < 2:
        return None
    
    # Find the best previous run (excluding the most recent)
    best_rmse = float('inf')
    baseline_run = None
    
    for run in runs[1:]:  # Skip most recent
        rmse = run.data.metrics.get('rmse', float('inf'))
        if rmse < best_rmse:
            best_rmse = rmse
            baseline_run = run
    
    return baseline_run


def generate_metrics_table(runs, baseline_run):
    """Generate markdown table comparing model metrics"""
    if not runs:
        return "No training runs found.\n"
    
    markdown = "## Model Performance Comparison\n\n"
    markdown += "| Model | RMSE | MAE | R² | MAPE | Status |\n"
    markdown += "|-------|------|-----|----|----- |--------|\n"
    
    baseline_rmse = baseline_run.data.metrics.get('rmse', float('inf')) if baseline_run else float('inf')
    
    for run in runs[:10]:  # Show top 10
        model_name = run.data.params.get('model_name', 'Unknown')
        rmse = run.data.metrics.get('rmse', 0)
        mae = run.data.metrics.get('mae', 0)
        r2 = run.data.metrics.get('r2', 0)
        mape = run.data.metrics.get('mape', 0)
        
        # Determine status
        if rmse < baseline_rmse * 0.95:
            status = "✅ Improved"
        elif rmse > baseline_rmse * 1.05:
            status = "⚠️ Degraded"
        else:
            status = "➡️ Similar"
        
        markdown += f"| {model_name} | {rmse:.4f} | {mae:.4f} | {r2:.4f} | {mape:.2f}% | {status} |\n"
    
    return markdown


def generate_improvement_analysis(runs, baseline_run):
    """Analyze improvement over baseline"""
    if not runs or not baseline_run:
        return ""
    
    latest_run = runs[0]
    
    baseline_rmse = baseline_run.data.metrics.get('rmse', float('inf'))
    latest_rmse = latest_run.data.metrics.get('rmse', float('inf'))
    
    baseline_r2 = baseline_run.data.metrics.get('r2', 0)
    latest_r2 = latest_run.data.metrics.get('r2', 0)
    
    rmse_change = ((latest_rmse - baseline_rmse) / baseline_rmse) * 100 if baseline_rmse > 0 else 0
    r2_change = latest_r2 - baseline_r2
    
    markdown = "\n## Performance Analysis\n\n"
    
    # RMSE Analysis
    if rmse_change < -5:
        markdown += f"🎉 **RMSE Improved by {abs(rmse_change):.2f}%** ({baseline_rmse:.4f} → {latest_rmse:.4f})\n\n"
    elif rmse_change > 5:
        markdown += f"⚠️ **RMSE Degraded by {rmse_change:.2f}%** ({baseline_rmse:.4f} → {latest_rmse:.4f})\n\n"
    else:
        markdown += f"➡️ **RMSE Stable** ({baseline_rmse:.4f} → {latest_rmse:.4f}, {rmse_change:+.2f}%)\n\n"
    
    # R² Analysis
    if r2_change > 0.01:
        markdown += f"📈 **R² Score Improved** ({baseline_r2:.4f} → {latest_r2:.4f}, {r2_change:+.4f})\n\n"
    elif r2_change < -0.01:
        markdown += f"📉 **R² Score Decreased** ({baseline_r2:.4f} → {latest_r2:.4f}, {r2_change:+.4f})\n\n"
    
    return markdown


def generate_model_summary(runs):
    """Generate summary of best models"""
    if not runs:
        return ""
    
    # Sort by RMSE
    sorted_runs = sorted(runs, key=lambda r: r.data.metrics.get('rmse', float('inf')))
    
    markdown = "\n## Best Models Summary\n\n"
    
    for i, run in enumerate(sorted_runs[:3], 1):
        model_name = run.data.params.get('model_name', 'Unknown')
        rmse = run.data.metrics.get('rmse', 0)
        r2 = run.data.metrics.get('r2', 0)
        
        medal = ['🥇', '🥈', '🥉'][i-1]
        markdown += f"{medal} **{model_name}**: RMSE={rmse:.4f}, R²={r2:.4f}\n\n"
    
    return markdown


def generate_training_details(runs):
    """Generate training run details"""
    if not runs:
        return ""
    
    latest_run = runs[0]
    
    markdown = "\n## Training Details\n\n"
    markdown += f"- **Run ID**: `{latest_run.info.run_id[:8]}...`\n"
    markdown += f"- **Model**: {latest_run.data.params.get('model_name', 'Unknown')}\n"
    markdown += f"- **Timestamp**: {datetime.fromtimestamp(latest_run.info.start_time/1000).strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    # Parameters
    markdown += "\n### Hyperparameters\n"
    for key, value in sorted(latest_run.data.params.items()):
        if key != 'model_name':
            markdown += f"- `{key}`: {value}\n"
    
    return markdown


def generate_recommendation(runs, baseline_run):
    """Generate deployment recommendation"""
    if not runs:
        return "\n## Recommendation\n\n⚠️ No training runs available for comparison.\n"
    
    latest_run = runs[0]
    latest_rmse = latest_run.data.metrics.get('rmse', float('inf'))
    
    if baseline_run:
        baseline_rmse = baseline_run.data.metrics.get('rmse', float('inf'))
        improvement = (baseline_rmse - latest_rmse) / baseline_rmse * 100
        
        if improvement > 5:
            return "\n## 🚀 Recommendation: DEPLOY\n\nThe new model shows significant improvement over baseline. Ready for production.\n"
        elif improvement > 0:
            return "\n## ✅ Recommendation: APPROVE\n\nThe new model shows marginal improvement. Safe to deploy.\n"
        elif improvement > -5:
            return "\n## ⚠️ Recommendation: REVIEW\n\nModel performance is similar to baseline. Manual review recommended.\n"
        else:
            return "\n## ❌ Recommendation: REJECT\n\nModel performance has degraded. Do not deploy without investigation.\n"
    
    return "\n## 📋 Recommendation: FIRST RUN\n\nThis appears to be the first training run. Baseline established.\n"


def main():
    """Generate complete CML report"""
    print("Generating CML Report...")
    
    # Setup MLflow
    client = setup_mlflow()
    
    # Get runs
    runs = get_recent_runs(client)
    baseline_run = get_baseline_metrics(runs)
    
    # Generate report sections
    report = "# 📊 Model Training Report\n\n"
    report += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
    report += generate_metrics_table(runs, baseline_run)
    report += generate_improvement_analysis(runs, baseline_run)
    report += generate_model_summary(runs)
    report += generate_training_details(runs)
    report += generate_recommendation(runs, baseline_run)
    
    # Add MLflow link
    report += "\n---\n"
    report += f"\n📈 [View Full Results in MLflow](https://dagshub.com/abdulhananch404/MLOPS_Project.mlflow)\n"
    
    # Output report
    output_file = os.getenv('CML_REPORT_FILE', 'cml_report.md')
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\n{report}")
    print(f"\n✅ Report saved to {output_file}")


if __name__ == "__main__":
    main()
