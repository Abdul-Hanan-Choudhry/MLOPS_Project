"""
Simple model training with MLflow logging to DagsHub.
Only logs params and metrics (no model artifacts) to verify DagsHub works.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

# Set credentials from environment or use defaults
dagshub_token = os.environ.get('DAGSHUB_TOKEN', '2f6456ef4e847038657172406b991bc3483f6c93')
os.environ['DAGSHUB_USERNAME'] = os.environ.get('MLFLOW_TRACKING_USERNAME', 'abdulhananch404')
os.environ['DAGSHUB_TOKEN'] = dagshub_token
os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('MLFLOW_TRACKING_USERNAME', 'abdulhananch404')
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# MLflow setup
MLFLOW_TRACKING_URI = "https://dagshub.com/abdulhananch404/MLOPS_Project.mlflow"

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def load_data():
    """Load the latest processed data."""
    processed_dir = "data/processed"
    parquet_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.parquet')])
    latest_file = os.path.join(processed_dir, parquet_files[-1])
    print(f"📂 Loading: {latest_file}")
    df = pd.read_parquet(latest_file)
    return df

def prepare_features(df):
    """Prepare features for training."""
    if 'target_price_1h' not in df.columns:
        df['target_price_1h'] = df['price'].shift(-1)
    
    df = df.dropna(subset=['target_price_1h'])
    
    exclude_cols = ['timestamp', 'target_price_1h', 'extraction_time']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    df = df.dropna(subset=feature_cols)
    
    X = df[feature_cols]
    y = df['target_price_1h']
    
    return X, y, feature_cols

def main():
    print("=" * 70)
    print("PHASE 2: MODEL TRAINING - SIMPLE VERSION")
    print("=" * 70)
    
    # Setup MLflow
    print(f"\n🔗 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get experiment
    experiment_name = "crypto-price-prediction"
    mlflow.set_experiment(experiment_name)
    print(f"✅ Using experiment: {experiment_name}")
    
    # Load data
    print("\n📊 Loading data...")
    df = load_data()
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X, y, feature_cols = prepare_features(df)
    print(f"   Features: {len(feature_cols)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define all 7 models
    models = {
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    }
    
    # Try to add XGBoost
    try:
        from xgboost import XGBRegressor
        models['xgboost'] = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
        print("✅ XGBoost available")
    except ImportError:
        print("⚠️ XGBoost not available")
    
    # Try to add LightGBM
    try:
        from lightgbm import LGBMRegressor
        models['lightgbm'] = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
        print("✅ LightGBM available")
    except ImportError:
        print("⚠️ LightGBM not available")
    
    print(f"\n🚀 Training {len(models)} models...")
    print("-" * 70)
    
    results = {}
    
    for model_name, model in models.items():
        try:
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                # Log basic parameters only
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("n_features", len(feature_cols))
                mlflow.log_param("n_train_samples", len(X_train))
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                mape = float(calculate_mape(y_test, y_pred))
                
                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mape", mape)
                
                # Set tags
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("stage", "training")
                
                results[model_name] = {
                    'model': model,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape
                }
                
                print(f"  ✅ {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
                
        except Exception as e:
            print(f"  ❌ {model_name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    if results:
        best_model_name = min(results.keys(), key=lambda k: results[k]['rmse'])
        best_result = results[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   RMSE: {best_result['rmse']:.4f}")
        print(f"   MAE: {best_result['mae']:.4f}")
        print(f"   R²: {best_result['r2']:.4f}")
        print(f"   MAPE: {best_result['mape']:.2f}%")
        
        # Save best model locally
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"best_model_{best_model_name}.joblib")
        joblib.dump(best_result['model'], model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"💾 Scaler saved to: {scaler_path}")
        
        print("\n" + "=" * 70)
        print("✅ PHASE 2 TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\n🔍 View experiments at:")
        print(f"   {MLFLOW_TRACKING_URI}/#/experiments/1")
    else:
        print("\n❌ No models trained successfully!")

if __name__ == "__main__":
    main()
