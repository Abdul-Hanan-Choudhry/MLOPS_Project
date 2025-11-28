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


def check_health():
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API")
        return False


def generate_normal_features():
    """Generate features within expected ranges."""
    return {
        "price": random.uniform(90000, 100000),
        "volume": random.uniform(4e10, 6e10),
        "market_cap": random.uniform(1.8e12, 2e12),
        "price_lag_1": random.uniform(90000, 100000),
        "price_lag_3": random.uniform(90000, 100000),
        "price_rolling_mean_6": random.uniform(90000, 100000),
        "price_rolling_std_6": random.uniform(100, 2000)
    }


def generate_ood_features():
    """Generate out-of-distribution features to trigger drift detection."""
    return {
        "price": random.uniform(50000, 70000),  # OOD - too low
        "volume": random.uniform(1e11, 2e11),   # OOD - too high
        "market_cap": random.uniform(1e12, 1.3e12),  # OOD - too low
        "price_lag_1": random.uniform(50000, 70000),
        "price_lag_3": random.uniform(50000, 70000),
        "price_rolling_mean_6": random.uniform(50000, 70000),
        "price_rolling_std_6": random.uniform(5000, 10000)  # OOD - too high
    }


def make_prediction(features):
    """Make a prediction request."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=5
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
                "prediction_request_latency_seconds",
                "data_drift_ood_ratio",
                "data_drift_detected",
                "model_predictions_total"
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
            print("\n🔍 Drift Status:")
            for feature, count in data.get("ood_counts", {}).items():
                total = data.get("total_counts", {}).get(feature, 0)
                ratio = count / total if total > 0 else 0
                print(f"  {feature}: {count}/{total} OOD ({ratio:.1%})")
            return True
        return False
    except Exception as e:
        print(f"Failed to get drift status: {e}")
        return False


def run_load_test(num_requests=50, ood_ratio=0.2):
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
            print(f"  [{i+1}/{num_requests}] {marker} Prediction: ${response.json()['prediction']:,.2f} ({latency*1000:.0f}ms)")
        else:
            error_count += 1
            print(f"  [{i+1}/{num_requests}] ❌ Failed")
        
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
