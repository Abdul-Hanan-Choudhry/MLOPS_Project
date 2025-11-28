"""
Prometheus Metrics Module for Crypto Price Prediction API.

Implements custom metrics for:
- API service monitoring (latency, request counts)
- Model/Data drift detection
- Feature distribution tracking
"""

import time
from typing import Dict, Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)
import numpy as np
from functools import wraps


# Create a custom registry for our metrics
REGISTRY = CollectorRegistry(auto_describe=True)

# ============================================================================
# SERVICE METRICS
# ============================================================================

# Request counter - total number of requests
REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

# Request latency histogram
REQUEST_LATENCY = Histogram(
    'prediction_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

# Active requests gauge
ACTIVE_REQUESTS = Gauge(
    'prediction_active_requests',
    'Number of active requests currently being processed',
    registry=REGISTRY
)

# ============================================================================
# MODEL METRICS
# ============================================================================

# Model predictions counter
PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total number of predictions made',
    ['model_name'],
    registry=REGISTRY
)

# Prediction value distribution
PREDICTION_VALUE = Histogram(
    'model_prediction_value',
    'Distribution of predicted values',
    ['model_name'],
    buckets=[50000, 60000, 70000, 80000, 90000, 95000, 100000, 105000, 110000, 120000, 150000],
    registry=REGISTRY
)

# Last prediction value
LAST_PREDICTION = Gauge(
    'model_last_prediction_value',
    'The most recent prediction value',
    ['model_name'],
    registry=REGISTRY
)

# ============================================================================
# DATA DRIFT METRICS
# ============================================================================

# Out-of-distribution ratio
OOD_RATIO = Gauge(
    'data_drift_ood_ratio',
    'Ratio of out-of-distribution feature values (drift proxy)',
    ['feature'],
    registry=REGISTRY
)

# Total OOD requests
OOD_REQUESTS_TOTAL = Counter(
    'data_drift_ood_requests_total',
    'Total requests with out-of-distribution features',
    registry=REGISTRY
)

# Feature value gauges for monitoring
FEATURE_VALUES = Gauge(
    'feature_value_current',
    'Current feature values for monitoring',
    ['feature'],
    registry=REGISTRY
)

# Feature statistics (for drift detection)
FEATURE_MEAN = Gauge(
    'feature_statistics_mean',
    'Running mean of feature values',
    ['feature'],
    registry=REGISTRY
)

FEATURE_STD = Gauge(
    'feature_statistics_std',
    'Running standard deviation of feature values',
    ['feature'],
    registry=REGISTRY
)

# Data drift flag
DATA_DRIFT_DETECTED = Gauge(
    'data_drift_detected',
    'Flag indicating if data drift is detected (1=yes, 0=no)',
    registry=REGISTRY
)

# Model info
MODEL_INFO = Info(
    'model',
    'Information about the loaded model',
    registry=REGISTRY
)


class DriftDetector:
    """
    Detects data drift by comparing incoming feature values against
    expected distributions.
    
    Uses a simple Z-score based approach as a drift proxy.
    """
    
    # Expected feature ranges based on crypto market data
    EXPECTED_RANGES = {
        'price': (80000, 120000),
        'volume': (3e10, 7e10),
        'market_cap': (1.5e12, 2.5e12),
        'price_lag_1': (80000, 120000),
        'price_lag_3': (80000, 120000),
        'price_rolling_mean_6': (80000, 120000),
        'price_rolling_std_6': (0, 5000),
    }
    
    # Running statistics for adaptive drift detection
    _feature_stats: Dict[str, Dict] = {}
    _ood_counts: Dict[str, int] = {}
    _total_counts: Dict[str, int] = {}
    _window_size: int = 1000
    
    @classmethod
    def check_features(cls, features: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if features are within expected ranges.
        
        Args:
            features: Dictionary of feature name to value
            
        Returns:
            Dictionary mapping feature names to OOD status (True if OOD)
        """
        ood_status = {}
        
        for name, value in features.items():
            is_ood = cls._check_single_feature(name, value)
            ood_status[name] = is_ood
            
            # Update running statistics
            cls._update_stats(name, value, is_ood)
            
            # Update Prometheus metrics
            FEATURE_VALUES.labels(feature=name).set(value)
            
            if name in cls._feature_stats:
                stats = cls._feature_stats[name]
                FEATURE_MEAN.labels(feature=name).set(stats.get('mean', 0))
                FEATURE_STD.labels(feature=name).set(stats.get('std', 0))
        
        # Calculate overall OOD ratio
        overall_ood = cls._calculate_overall_ood()
        DATA_DRIFT_DETECTED.set(1 if overall_ood > 0.1 else 0)
        
        return ood_status
    
    @classmethod
    def _check_single_feature(cls, name: str, value: float) -> bool:
        """Check if a single feature value is out of distribution."""
        # Check against expected ranges
        if name in cls.EXPECTED_RANGES:
            min_val, max_val = cls.EXPECTED_RANGES[name]
            if value < min_val or value > max_val:
                return True
        
        # Check against running statistics (Z-score > 3)
        if name in cls._feature_stats:
            stats = cls._feature_stats[name]
            mean = stats.get('mean', 0)
            std = stats.get('std', 1)
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > 3:
                    return True
        
        return False
    
    @classmethod
    def _update_stats(cls, name: str, value: float, is_ood: bool):
        """Update running statistics for a feature."""
        # Initialize if needed
        if name not in cls._feature_stats:
            cls._feature_stats[name] = {
                'values': [],
                'mean': value,
                'std': 0
            }
            cls._ood_counts[name] = 0
            cls._total_counts[name] = 0
        
        # Update counts
        cls._total_counts[name] += 1
        if is_ood:
            cls._ood_counts[name] += 1
            OOD_REQUESTS_TOTAL.inc()
        
        # Update running statistics (using exponential moving average)
        stats = cls._feature_stats[name]
        alpha = 0.01  # Smoothing factor
        
        old_mean = stats['mean']
        stats['mean'] = (1 - alpha) * old_mean + alpha * value
        stats['std'] = np.sqrt((1 - alpha) * stats['std']**2 + alpha * (value - old_mean)**2)
        
        # Update OOD ratio metric
        if cls._total_counts[name] > 0:
            ood_ratio = cls._ood_counts[name] / cls._total_counts[name]
            OOD_RATIO.labels(feature=name).set(ood_ratio)
    
    @classmethod
    def _calculate_overall_ood(cls) -> float:
        """Calculate overall OOD ratio across all features."""
        total_ood = sum(cls._ood_counts.values())
        total_requests = sum(cls._total_counts.values())
        
        if total_requests == 0:
            return 0.0
        
        return total_ood / total_requests
    
    @classmethod
    def reset_stats(cls):
        """Reset all statistics (useful for testing)."""
        cls._feature_stats = {}
        cls._ood_counts = {}
        cls._total_counts = {}


class MetricsMiddleware:
    """
    Middleware to automatically track request metrics.
    """
    
    @staticmethod
    def track_request(method: str, endpoint: str, status: str, duration: float):
        """Track a completed request."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
    
    @staticmethod
    def track_prediction(model_name: str, prediction_value: float):
        """Track a prediction."""
        PREDICTIONS_TOTAL.labels(model_name=model_name).inc()
        PREDICTION_VALUE.labels(model_name=model_name).observe(prediction_value)
        LAST_PREDICTION.labels(model_name=model_name).set(prediction_value)
    
    @staticmethod
    def set_model_info(name: str, model_type: str, version: str):
        """Set model information."""
        MODEL_INFO.info({
            'name': name,
            'type': model_type,
            'version': version
        })


def get_metrics():
    """Generate metrics in Prometheus format."""
    return generate_latest(REGISTRY)


def get_content_type():
    """Get the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST
