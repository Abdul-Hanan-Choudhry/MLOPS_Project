"""
Helper utility functions for the MLOps project
"""
from datetime import datetime
from typing import Any, Dict


def get_timestamp() -> str:
    """Get current timestamp in standard format"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary for display"""
    lines = []
    for name, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"  {name}: {value:.4f}")
        else:
            lines.append(f"  {name}: {value}")
    return "\n".join(lines)


def validate_positive(value: Any, name: str) -> float:
    """Validate that a value is positive"""
    try:
        num = float(value)
        if num <= 0:
            raise ValueError(f"{name} must be positive, got {num}")
        return num
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid {name}: {e}")
