#!/usr/bin/env python
"""
Test script for monitoring stack.

Sends sample requests to the API to generate metrics for Prometheus/Grafana.
Run this after starting the monitoring stack with docker-compose.

Usage:
    python scripts/test_monitoring.py
"""

import requests
import time
import random
import json
from datetime import datetime


API_URL = "http://localhost:8000"

# Feature names expected by the model (from the scaler)
FEATURE_NAMES = [
    'price', 'market_cap', 'volume', 'price_lag_1h', 'volume_lag_1h',
    'price_change_1h', 'price_pct_change_1h', 'price_lag_3h', 'volume_lag_3h',
    'price_change_3h', 'price_pct_change_3h', 'price_lag_6h', 'volume_lag_6h',
    'price_change_6h', 'price_pct_change_6h', 'price_lag_12h', 'volume_lag_12h',
    'price_change_12h', 'price_pct_change_12h', 'price_lag_24h',
    'volume_lag_24h', 'price_change_24h', 'price_pct_change_24h',
    'price_rolling_mean_6h', 'price_rolling_std_6h', 'price_rolling_min_6h',
    'price_rolling_max_6h', 'volume_rolling_mean_6h', 'volume_rolling_std_6h',
    'market_cap_rolling_mean_6h', 'price_rolling_mean_12h',
    'price_rolling_std_12h', 'price_rolling_min_12h', 'price_rolling_max_12h',
    'volume_rolling_mean_12h', 'volume_rolling_std_12h',
    'market_cap_rolling_mean_12h', 'price_rolling_mean_24h',
    'price_rolling_std_24h', 'price_rolling_min_24h', 'price_rolling_max_24h',
    'volume_rolling_mean_24h', 'volume_rolling_std_24h',
    'market_cap_rolling_mean_24h', 'hour', 'day_of_week', 'day_of_month', 'month',
    'week_of_year', 'is_weekend', 'is_night', 'is_trading_hours', 'hour_sin',
    'hour_cos', 'dow_sin', 'dow_cos', 'volatility_6h', 'volatility_12h',
    'volatility_24h', 'momentum_12h', 'momentum_24h', 'price_position_6h',
    'price_position_12h', 'price_position_24h', 'volume_ratio_24h',
    'pv_momentum', 'target_pct_change_1h', 'target_direction_1h'
]


def check_health():
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy - Model: {data.get('model_name', 'N/A')}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API")
        return False


def generate_normal_features():
    """Generate features within expected ranges (Bitcoin-like values)."""
    base_price = random.uniform(90000, 100000)
    base_volume = random.uniform(4e10, 6e10)
    base_market_cap = random.uniform(1.8e12, 2e12)
    
    features = {
        'price': base_price,
        'market_cap': base_market_cap,
        'volume': base_volume,
        
        # Lag features (slightly varied from base)
        'price_lag_1h': base_price * random.uniform(0.995, 1.005),
        'volume_lag_1h': base_volume * random.uniform(0.9, 1.1),
        'price_change_1h': random.uniform(-500, 500),
        'price_pct_change_1h': random.uniform(-0.005, 0.005),
        
        'price_lag_3h': base_price * random.uniform(0.99, 1.01),
        'volume_lag_3h': base_volume * random.uniform(0.85, 1.15),
        'price_change_3h': random.uniform(-1000, 1000),
        'price_pct_change_3h': random.uniform(-0.01, 0.01),
        
        'price_lag_6h': base_price * random.uniform(0.98, 1.02),
        'volume_lag_6h': base_volume * random.uniform(0.8, 1.2),
        'price_change_6h': random.uniform(-1500, 1500),
        'price_pct_change_6h': random.uniform(-0.015, 0.015),
        
        'price_lag_12h': base_price * random.uniform(0.97, 1.03),
        'volume_lag_12h': base_volume * random.uniform(0.75, 1.25),
        'price_change_12h': random.uniform(-2000, 2000),
        'price_pct_change_12h': random.uniform(-0.02, 0.02),
        
        'price_lag_24h': base_price * random.uniform(0.95, 1.05),
        'volume_lag_24h': base_volume * random.uniform(0.7, 1.3),
        'price_change_24h': random.uniform(-3000, 3000),
        'price_pct_change_24h': random.uniform(-0.03, 0.03),
        
        # Rolling statistics
        'price_rolling_mean_6h': base_price * random.uniform(0.99, 1.01),
        'price_rolling_std_6h': random.uniform(100, 500),
        'price_rolling_min_6h': base_price * 0.98,
        'price_rolling_max_6h': base_price * 1.02,
        'volume_rolling_mean_6h': base_volume,
        'volume_rolling_std_6h': base_volume * 0.1,
        'market_cap_rolling_mean_6h': base_market_cap,
        
        'price_rolling_mean_12h': base_price * random.uniform(0.98, 1.02),
        'price_rolling_std_12h': random.uniform(150, 600),
        'price_rolling_min_12h': base_price * 0.97,
        'price_rolling_max_12h': base_price * 1.03,
        'volume_rolling_mean_12h': base_volume,
        'volume_rolling_std_12h': base_volume * 0.12,
        'market_cap_rolling_mean_12h': base_market_cap,
        
        'price_rolling_mean_24h': base_price * random.uniform(0.97, 1.03),
        'price_rolling_std_24h': random.uniform(200, 800),
        'price_rolling_min_24h': base_price * 0.95,
        'price_rolling_max_24h': base_price * 1.05,
        'volume_rolling_mean_24h': base_volume,
        'volume_rolling_std_24h': base_volume * 0.15,
        'market_cap_rolling_mean_24h': base_market_cap,
        
        # Time features
        'hour': random.randint(0, 23),
        'day_of_week': random.randint(0, 6),
        'day_of_month': random.randint(1, 28),
        'month': random.randint(1, 12),
        'week_of_year': random.randint(1, 52),
        'is_weekend': random.choice([0, 1]),
        'is_night': random.choice([0, 1]),
        'is_trading_hours': random.choice([0, 1]),
        'hour_sin': random.uniform(-1, 1),
        'hour_cos': random.uniform(-1, 1),
        'dow_sin': random.uniform(-1, 1),
        'dow_cos': random.uniform(-1, 1),
        
        # Volatility and momentum
        'volatility_6h': random.uniform(0.001, 0.01),
        'volatility_12h': random.uniform(0.002, 0.015),
        'volatility_24h': random.uniform(0.003, 0.02),
        'momentum_12h': random.uniform(-0.02, 0.02),
        'momentum_24h': random.uniform(-0.03, 0.03),
        
        # Position features
        'price_position_6h': random.uniform(0, 1),
        'price_position_12h': random.uniform(0, 1),
        'price_position_24h': random.uniform(0, 1),
        'volume_ratio_24h': random.uniform(0.8, 1.2),
        'pv_momentum': random.uniform(-0.05, 0.05),
        
        # Target features (for inference)
        'target_pct_change_1h': random.uniform(-0.01, 0.01),
        'target_direction_1h': random.choice([0, 1])
    }
    
    return features


def generate_ood_features():
    """Generate out-of-distribution features to trigger drift detection."""
    # Start with normal features
    features = generate_normal_features()
    
    # Make some features out of distribution
    features['price'] = random.uniform(50000, 70000)  # OOD - too low
    features['volume'] = random.uniform(1e11, 2e11)   # OOD - too high
    features['market_cap'] = random.uniform(1e12, 1.3e12)  # OOD - too low
    features['volatility_24h'] = random.uniform(0.1, 0.3)  # OOD - extremely high
    features['momentum_24h'] = random.uniform(-0.2, 0.2)   # OOD - extreme momentum
    
    return features


def make_prediction(features):
    """Make a prediction request."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=10
        )
        return response
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def check_metrics():
    """Retrieve and display metrics."""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            # Parse key metrics
            lines = response.text.split("\n")
            key_metrics = [
                "prediction_requests_total",
                "data_drift_ood_ratio",
                "data_drift_detected",
                "model_predictions_total",
                "model_last_prediction"
            ]
            
            print("\n📊 Key Metrics:")
            for line in lines:
                for metric in key_metrics:
                    if line.startswith(metric) and not line.startswith("#"):
                        print(f"  {line}")
            return True
        return False
    except Exception as e:
        print(f"Failed to get metrics: {e}")
        return False


def check_drift_status():
    """Check drift detection status."""
    try:
        response = requests.get(f"{API_URL}/drift/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            ood_counts = data.get("ood_counts", {})
            total_counts = data.get("total_counts", {})
            
            if ood_counts:
                print("\n🔍 Drift Status (top features):")
                # Show only a few key features
                key_features = ['price', 'volume', 'market_cap', 'volatility_24h', 'momentum_24h']
                for feature in key_features:
                    if feature in ood_counts:
                        count = ood_counts[feature]
                        total = total_counts.get(feature, 0)
                        ratio = count / total if total > 0 else 0
                        print(f"  {feature}: {count}/{total} OOD ({ratio:.1%})")
            else:
                print("\n🔍 Drift Status: No data yet")
            return True
        return False
    except Exception as e:
        print(f"Failed to get drift status: {e}")
        return False


def run_load_test(num_requests=30, ood_ratio=0.2):
    """
    Run a load test with mixed normal and OOD requests.
    
    Args:
        num_requests: Total number of requests to send
        ood_ratio: Ratio of OOD requests (0.0 to 1.0)
    """
    print(f"\n🚀 Starting load test with {num_requests} requests ({ood_ratio:.0%} OOD)...")
    
    success_count = 0
    error_count = 0
    latencies = []
    
    for i in range(num_requests):
        # Decide if this should be OOD
        is_ood = random.random() < ood_ratio
        features = generate_ood_features() if is_ood else generate_normal_features()
        
        start_time = time.time()
        response = make_prediction(features)
        latency = time.time() - start_time
        
        if response and response.status_code == 200:
            success_count += 1
            latencies.append(latency)
            marker = "⚠️ OOD" if is_ood else "✓"
            prediction = response.json().get('prediction', 0)
            print(f"  [{i+1}/{num_requests}] {marker} Prediction: ${prediction:,.2f} ({latency*1000:.0f}ms)")
        else:
            error_count += 1
            status = response.status_code if response else "N/A"
            detail = ""
            if response:
                try:
                    detail = response.json().get('detail', '')[:50]
                except:
                    pass
            print(f"  [{i+1}/{num_requests}] ❌ Failed (HTTP {status}) {detail}")
        
        # Small delay between requests
        time.sleep(0.1)
    
    # Summary
    print(f"\n📈 Load Test Summary:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    if latencies:
        print(f"  Avg latency: {sum(latencies)/len(latencies)*1000:.0f}ms")
        print(f"  Max latency: {max(latencies)*1000:.0f}ms")


def main():
    """Main test runner."""
    print("=" * 60)
    print("🔬 Crypto Price Prediction - Monitoring Test")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"API URL: {API_URL}")
    print()
    
    # Check API health
    if not check_health():
        print("\n⚠️ API is not running. Start the monitoring stack first:")
        print("  cd monitoring")
        print("  docker-compose -f docker-compose.monitoring.yml up -d")
        return
    
    # Run load test
    run_load_test(num_requests=30, ood_ratio=0.3)
    
    # Check metrics
    check_metrics()
    
    # Check drift status
    check_drift_status()
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print()
    print("📊 View dashboards at:")
    print("  - Grafana: http://localhost:3000 (admin/admin)")
    print("  - Prometheus: http://localhost:9090")
    print("  - API Docs: http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
