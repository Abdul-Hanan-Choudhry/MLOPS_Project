"""
Run model training locally with MLflow logging to DagsHub.
This script verifies Phase 2 implementation.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

# Set credentials BEFORE importing mlflow
os.environ['DAGSHUB_USERNAME'] = 'abdulhananch404'
os.environ['DAGSHUB_TOKEN'] = '2f6456ef4e847038657172406b991bc3483f6c93'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'abdulhananch404'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '2f6456ef4e847038657172406b991bc3483f6c93'

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
import json

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
    
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Directory {processed_dir} not found!")
    
    parquet_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.parquet')])
    
    if not parquet_files:
        raise FileNotFoundError("No parquet files found in data/processed/")
    
    latest_file = os.path.join(processed_dir, parquet_files[-1])
    print(f"📂 Loading: {latest_file}")
    
    df = pd.read_parquet(latest_file)
    return df

def prepare_features(df):
    """Prepare features for training."""
    # Create target if not exists
    if 'target_price_1h' not in df.columns:
        df['target_price_1h'] = df['price'].shift(-1)
    
    # Drop rows with NaN target
    df = df.dropna(subset=['target_price_1h'])
    
    # Select feature columns (exclude non-numeric and target)
    exclude_cols = ['timestamp', 'target_price_1h', 'extraction_time']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    # Drop any remaining NaN
    df = df.dropna(subset=feature_cols)
    
    X = df[feature_cols]
    y = df['target_price_1h']
    
    return X, y, feature_cols

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, feature_names):
    """Train a model and log to MLflow."""
    
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        params = model.get_params()
        for param_name, param_value in params.items():
            try:
                mlflow.log_param(param_name, param_value)
            except:
                pass
        
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Set tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("stage", "training")
        
        print(f"  ✅ {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
        
        return {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }

def main():
    print("=" * 70)
    print("PHASE 2: MODEL TRAINING WITH MLFLOW TRACKING")
    print("=" * 70)
    
    # Setup MLflow
    print(f"\n🔗 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get experiment
    experiment_name = "crypto-price-prediction"
    try:
        mlflow.set_experiment(experiment_name)
        print(f"✅ Using experiment: {experiment_name}")
    except Exception as e:
        print(f"⚠️ Experiment setup: {e}")
    
    # Load data
    print("\n📊 Loading data...")
    df = load_data()
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X, y, feature_names = prepare_features(df)
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle for time series
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }
    
    # Try to add XGBoost and LightGBM
    try:
        from xgboost import XGBRegressor
        models['xgboost'] = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0, n_jobs=-1)
        print("✅ XGBoost available")
    except ImportError:
        print("⚠️ XGBoost not installed, skipping...")
    
    try:
        from lightgbm import LGBMRegressor
        models['lightgbm'] = LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1, n_jobs=-1)
        print("✅ LightGBM available")
    except ImportError:
        print("⚠️ LightGBM not installed, skipping...")
    
    # Train all models
    print(f"\n🚀 Training {len(models)} models...")
    print("-" * 70)
    
    results = {}
    for model_name, model in models.items():
        try:
            # Use scaled data for linear models
            if model_name in ['ridge', 'lasso', 'elasticnet']:
                result = train_and_log_model(
                    model, model_name, 
                    X_train_scaled, X_test_scaled, 
                    y_train, y_test, 
                    feature_names
                )
            else:
                result = train_and_log_model(
                    model, model_name,
                    X_train.values, X_test.values,
                    y_train.values, y_test.values,
                    feature_names
                )
            results[model_name] = result
        except Exception as e:
            print(f"  ❌ {model_name}: {e}")
    
    # Find best model
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    if not results:
        print("❌ No models trained successfully!")
        return
    
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
    print(f"   RMSE: {best_result['rmse']:.2f}")
    print(f"   MAE:  {best_result['mae']:.2f}")
    print(f"   R²:   {best_result['r2']:.4f}")
    print(f"   MAPE: {best_result['mape']:.2f}%")
    
    # Print all results
    print(f"\n📊 All Model Results:")
    print("-" * 70)
    print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'MAPE':<12}")
    print("-" * 70)
    for model_name, result in sorted(results.items(), key=lambda x: x[1]['rmse']):
        print(f"{model_name:<20} {result['rmse']:<12.2f} {result['mae']:<12.2f} {result['r2']:<12.4f} {result['mape']:<12.2f}%")
    
    # Save best model locally
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_model.pkl'
    joblib.dump(best_result['model'], model_path)
    print(f"\n💾 Best model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'metrics': {
            'rmse': float(best_result['rmse']),
            'mae': float(best_result['mae']),
            'r2': float(best_result['r2']),
            'mape': float(best_result['mape'])
        },
        'features': feature_names,
        'trained_at': datetime.now().isoformat()
    }
    metadata_path = 'models/best_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadata saved: {metadata_path}")
    
    # Save scaler
    scaler_path = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"📐 Scaler saved: {scaler_path}")
    
    # Register best model to MLflow Model Registry
    print("\n📦 Registering model to MLflow Model Registry...")
    try:
        with mlflow.start_run(run_name=f"best_{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("model_type", best_model_name)
            mlflow.log_param("is_best_model", True)
            for metric_name, metric_value in best_result.items():
                if metric_name != 'model':
                    mlflow.log_metric(metric_name, metric_value)
            
            mlflow.set_tag("best_model", "true")
            mlflow.set_tag("stage", "production_candidate")
            
            # Log model (without registration to avoid unsupported endpoint error)
            mlflow.sklearn.log_model(best_result['model'], "model")
        print("✅ Best model logged to MLflow")
    except Exception as e:
        print(f"⚠️ Best model logging: {e}")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 2 COMPLETE!")
    print("=" * 70)
    print(f"\n🔗 View experiments at:")
    print(f"   {MLFLOW_TRACKING_URI}")
    print(f"\n📁 Local artifacts:")
    print(f"   models/best_model.pkl")
    print(f"   models/best_model_metadata.json")
    print(f"   models/scaler.pkl")

if __name__ == "__main__":
    main()
