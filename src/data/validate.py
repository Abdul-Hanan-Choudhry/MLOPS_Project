"""
Data Quality Validation Module
Implements strict data quality checks as a mandatory gate in the pipeline.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import data_quality_config

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # error, warning, info


@dataclass
class QualityReport:
    """Complete quality validation report."""
    passed: bool
    timestamp: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[ValidationResult]
    data_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                    "severity": r.severity
                }
                for r in self.results
            ],
            "data_stats": self.data_stats
        }


class DataQualityValidator:
    """
    Validates data quality with configurable checks.
    
    Implements mandatory quality gates:
    - Null value checks (>1% threshold fails)
    - Schema validation
    - Data freshness checks
    - Value range validation
    """
    
    def __init__(
        self,
        null_threshold: float = None,
        min_rows: int = None,
        required_columns: List[str] = None
    ):
        """
        Initialize the validator.
        
        Args:
            null_threshold: Maximum allowed null ratio (0.01 = 1%)
            min_rows: Minimum required number of rows
            required_columns: List of columns that must be present
        """
        self.null_threshold = null_threshold or data_quality_config.NULL_THRESHOLD
        self.min_rows = min_rows or data_quality_config.MIN_ROWS_REQUIRED
        self.required_columns = required_columns or data_quality_config.REQUIRED_COLUMNS
        
        logger.info(
            f"Initialized DataQualityValidator: "
            f"null_threshold={self.null_threshold}, "
            f"min_rows={self.min_rows}, "
            f"required_columns={self.required_columns}"
        )
    
    def check_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate that all required columns are present.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with check outcome
        """
        missing_columns = set(self.required_columns) - set(df.columns)
        passed = len(missing_columns) == 0
        
        return ValidationResult(
            check_name="schema_validation",
            passed=passed,
            message="All required columns present" if passed else f"Missing columns: {missing_columns}",
            details={
                "required_columns": self.required_columns,
                "actual_columns": list(df.columns),
                "missing_columns": list(missing_columns)
            },
            severity="error"
        )
    
    def check_null_values(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check that null values don't exceed threshold in key columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with check outcome
        """
        # Check only columns that exist
        columns_to_check = [c for c in self.required_columns if c in df.columns]
        
        if not columns_to_check:
            return ValidationResult(
                check_name="null_value_check",
                passed=False,
                message="No columns to check for null values",
                details={},
                severity="error"
            )
        
        null_ratios = df[columns_to_check].isnull().mean().to_dict()
        failed_columns = {
            col: ratio for col, ratio in null_ratios.items()
            if ratio > self.null_threshold
        }
        
        passed = len(failed_columns) == 0
        
        return ValidationResult(
            check_name="null_value_check",
            passed=passed,
            message=f"Null check passed (threshold: {self.null_threshold*100}%)" if passed 
                    else f"Null threshold exceeded in columns: {list(failed_columns.keys())}",
            details={
                "threshold": self.null_threshold,
                "null_ratios": null_ratios,
                "failed_columns": failed_columns
            },
            severity="error"
        )
    
    def check_row_count(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate minimum row count requirement.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with check outcome
        """
        row_count = len(df)
        passed = row_count >= self.min_rows
        
        return ValidationResult(
            check_name="row_count_check",
            passed=passed,
            message=f"Row count: {row_count} (min required: {self.min_rows})",
            details={
                "row_count": row_count,
                "min_required": self.min_rows
            },
            severity="error"
        )
    
    def check_data_freshness(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "timestamp",
        max_age_hours: int = 24
    ) -> ValidationResult:
        """
        Validate that data is recent enough.
        
        Args:
            df: DataFrame to validate
            timestamp_column: Column containing timestamps
            max_age_hours: Maximum acceptable age of latest record
            
        Returns:
            ValidationResult with check outcome
        """
        if timestamp_column not in df.columns:
            return ValidationResult(
                check_name="data_freshness_check",
                passed=False,
                message=f"Timestamp column '{timestamp_column}' not found",
                details={},
                severity="warning"
            )
        
        latest_timestamp = pd.to_datetime(df[timestamp_column]).max()
        current_time = datetime.utcnow()
        age_hours = (current_time - latest_timestamp).total_seconds() / 3600
        
        passed = age_hours <= max_age_hours
        
        return ValidationResult(
            check_name="data_freshness_check",
            passed=passed,
            message=f"Data age: {age_hours:.1f} hours (max allowed: {max_age_hours}h)",
            details={
                "latest_timestamp": str(latest_timestamp),
                "current_time": str(current_time),
                "age_hours": round(age_hours, 2),
                "max_age_hours": max_age_hours
            },
            severity="warning"
        )
    
    def check_value_ranges(
        self,
        df: pd.DataFrame,
        column_ranges: Dict[str, tuple] = None
    ) -> ValidationResult:
        """
        Validate that values are within expected ranges.
        
        Args:
            df: DataFrame to validate
            column_ranges: Dict mapping column names to (min, max) tuples
            
        Returns:
            ValidationResult with check outcome
        """
        if column_ranges is None:
            # Default ranges for crypto data
            column_ranges = {
                "price": (0, float("inf")),
                "volume": (0, float("inf")),
                "market_cap": (0, float("inf"))
            }
        
        out_of_range = {}
        for col, (min_val, max_val) in column_ranges.items():
            if col not in df.columns:
                continue
            
            below_min = (df[col] < min_val).sum()
            above_max = (df[col] > max_val).sum() if max_val != float("inf") else 0
            
            if below_min > 0 or above_max > 0:
                out_of_range[col] = {
                    "below_min": int(below_min),
                    "above_max": int(above_max),
                    "range": (min_val, max_val)
                }
        
        passed = len(out_of_range) == 0
        
        return ValidationResult(
            check_name="value_range_check",
            passed=passed,
            message="All values within expected ranges" if passed
                    else f"Values out of range in columns: {list(out_of_range.keys())}",
            details={
                "expected_ranges": {k: v for k, v in column_ranges.items() if k in df.columns},
                "out_of_range_columns": out_of_range
            },
            severity="error"
        )
    
    def check_duplicates(
        self,
        df: pd.DataFrame,
        subset: List[str] = None
    ) -> ValidationResult:
        """
        Check for duplicate records.
        
        Args:
            df: DataFrame to validate
            subset: Columns to use for duplicate detection
            
        Returns:
            ValidationResult with check outcome
        """
        if subset is None:
            subset = ["timestamp"] if "timestamp" in df.columns else None
        
        duplicate_count = df.duplicated(subset=subset).sum()
        passed = duplicate_count == 0
        
        return ValidationResult(
            check_name="duplicate_check",
            passed=passed,
            message=f"Found {duplicate_count} duplicate records" if not passed
                    else "No duplicate records found",
            details={
                "duplicate_count": int(duplicate_count),
                "total_rows": len(df),
                "duplicate_percentage": round(duplicate_count / len(df) * 100, 2) if len(df) > 0 else 0
            },
            severity="warning"
        )
    
    def compute_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute summary statistics for the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats["numeric_stats"] = df[numeric_cols].describe().to_dict()
        
        # Timestamp range
        if "timestamp" in df.columns:
            stats["timestamp_range"] = {
                "min": str(df["timestamp"].min()),
                "max": str(df["timestamp"].max())
            }
        
        return stats
    
    def validate(
        self,
        df: pd.DataFrame,
        include_warnings: bool = True
    ) -> QualityReport:
        """
        Run all validation checks on the DataFrame.
        
        Args:
            df: DataFrame to validate
            include_warnings: Whether to include warning-level checks
            
        Returns:
            QualityReport with all check results
        """
        logger.info(f"Starting data quality validation on {len(df)} rows")
        
        results = []
        
        # Run all checks
        results.append(self.check_schema(df))
        results.append(self.check_null_values(df))
        results.append(self.check_row_count(df))
        results.append(self.check_value_ranges(df))
        results.append(self.check_duplicates(df))
        
        if include_warnings:
            results.append(self.check_data_freshness(df))
        
        # Compute statistics
        data_stats = self.compute_data_stats(df)
        
        # Determine overall pass/fail (only error severity fails the pipeline)
        error_results = [r for r in results if r.severity == "error"]
        overall_passed = all(r.passed for r in error_results)
        
        passed_count = sum(1 for r in results if r.passed)
        failed_count = sum(1 for r in results if not r.passed)
        
        report = QualityReport(
            passed=overall_passed,
            timestamp=datetime.utcnow(),
            total_checks=len(results),
            passed_checks=passed_count,
            failed_checks=failed_count,
            results=results,
            data_stats=data_stats
        )
        
        # Log results
        for result in results:
            if result.passed:
                logger.info(f"✓ {result.check_name}: {result.message}")
            else:
                if result.severity == "error":
                    logger.error(f"✗ {result.check_name}: {result.message}")
                else:
                    logger.warning(f"⚠ {result.check_name}: {result.message}")
        
        logger.info(
            f"Validation complete: {passed_count}/{len(results)} checks passed. "
            f"Overall: {'PASSED' if overall_passed else 'FAILED'}"
        )
        
        return report
    
    def validate_or_fail(self, df: pd.DataFrame) -> QualityReport:
        """
        Run validation and raise exception if quality check fails.
        
        This is the mandatory quality gate that stops the pipeline
        if data quality is insufficient.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            QualityReport if validation passes
            
        Raises:
            DataQualityError: If validation fails
        """
        report = self.validate(df)
        
        if not report.passed:
            failed_checks = [r for r in report.results if not r.passed and r.severity == "error"]
            error_messages = [f"{r.check_name}: {r.message}" for r in failed_checks]
            
            raise DataQualityError(
                f"Data quality validation failed. "
                f"Failed checks: {error_messages}",
                report=report
            )
        
        return report


class DataQualityError(Exception):
    """Exception raised when data quality validation fails."""
    
    def __init__(self, message: str, report: QualityReport = None):
        super().__init__(message)
        self.report = report


def validate_data(
    df: pd.DataFrame,
    fail_on_error: bool = True,
    **kwargs
) -> QualityReport:
    """
    Main validation function for use in pipeline.
    
    Args:
        df: DataFrame to validate
        fail_on_error: Whether to raise exception on failure
        **kwargs: Additional arguments for validator
        
    Returns:
        QualityReport
    """
    validator = DataQualityValidator(**kwargs)
    
    if fail_on_error:
        return validator.validate_or_fail(df)
    else:
        return validator.validate(df)


if __name__ == "__main__":
    # Test the validator
    logger.info("Running data quality validator test...")
    
    # Create sample data
    import numpy as np
    
    dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
    test_df = pd.DataFrame({
        "timestamp": dates,
        "price": np.random.uniform(40000, 50000, 100),
        "volume": np.random.uniform(1e9, 2e9, 100),
        "market_cap": np.random.uniform(8e11, 9e11, 100)
    })
    
    # Add some nulls for testing
    test_df.loc[0, "price"] = None
    
    print(f"\n{'='*50}")
    print("Data Quality Validation Test")
    print(f"{'='*50}")
    
    validator = DataQualityValidator()
    report = validator.validate(test_df)
    
    print(f"\nOverall Result: {'PASSED' if report.passed else 'FAILED'}")
    print(f"Checks: {report.passed_checks}/{report.total_checks} passed")
    
    print(f"\nDetailed Results:")
    for result in report.results:
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.check_name}: {result.message}")
