"""
MLflow Model Registry Module
Manages model versioning, staging, and deployment through MLflow Model Registry.

This module:
- Registers trained models to MLflow Model Registry
- Manages model versions and stages (Staging, Production, Archived)
- Promotes/demotes models between stages
- Retrieves production-ready models for inference
- Integrates with DagsHub as remote registry
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import MODELS_DIR

logger = get_logger(__name__)


class ModelRegistry:
    """
    Manages MLflow Model Registry operations.
    
    Provides functionality to:
    - Register new models
    - Transition model stages
    - Retrieve models by stage
    - Compare model versions
    - Clean up old versions
    """
    
    # Valid model stages
    STAGES = ["None", "Staging", "Production", "Archived"]
    
    def __init__(
        self,
        tracking_uri: str = None,
        registry_uri: str = None
    ):
        """
        Initialize the Model Registry manager.
        
        Args:
            tracking_uri: MLflow tracking URI
            registry_uri: MLflow registry URI (usually same as tracking)
        """
        self._setup_mlflow(tracking_uri, registry_uri)
        self.client = MlflowClient()
        
        logger.info("Initialized ModelRegistry")
        logger.info(f"  Tracking URI: {mlflow.get_tracking_uri()}")
    
    def _setup_mlflow(
        self, 
        tracking_uri: str = None,
        registry_uri: str = None
    ):
        """Configure MLflow with DagsHub."""
        dagshub_username = os.getenv("DAGSHUB_USERNAME", "")
        dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
        mlflow_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
        
        if dagshub_username and dagshub_token and mlflow_uri:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            
            mlflow.set_tracking_uri(mlflow_uri)
            if registry_uri:
                mlflow.set_registry_uri(registry_uri)
            
            logger.info(f"Connected to DagsHub MLflow: {mlflow_uri}")
        else:
            local_uri = f"file://{MODELS_DIR}/mlruns"
            mlflow.set_tracking_uri(local_uri)
            logger.warning(f"Using local MLflow: {local_uri}")
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        description: str = None,
        tags: Dict[str, str] = None
    ) -> str:
        """
        Register a model from an MLflow run to the Model Registry.
        
        Args:
            run_id: MLflow run ID containing the model
            model_name: Name for the registered model
            artifact_path: Path to model artifact in run
            description: Model description
            tags: Additional tags for the model version
            
        Returns:
            Model version number
        """
        logger.info(f"Registering model '{model_name}' from run {run_id}")
        
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        try:
            # Register the model
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            version = result.version
            logger.info(f"Registered model version: {version}")
            
            # Add description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description
                )
            
            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=key,
                        value=value
                    )
            
            return version
            
        except MlflowException as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ) -> bool:
        """
        Transition a model version to a new stage.
        
        Args:
            model_name: Name of registered model
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Archive existing models in target stage
            
        Returns:
            True if successful
        """
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {self.STAGES}")
        
        logger.info(f"Transitioning {model_name} v{version} to {stage}")
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            
            logger.info(f"Successfully transitioned to {stage}")
            return True
            
        except MlflowException as e:
            logger.error(f"Failed to transition stage: {e}")
            return False
    
    def promote_to_staging(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """Promote a model version to Staging."""
        return self.transition_model_stage(
            model_name=model_name,
            version=version,
            stage="Staging"
        )
    
    def promote_to_production(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """Promote a model version to Production."""
        return self.transition_model_stage(
            model_name=model_name,
            version=version,
            stage="Production",
            archive_existing=True
        )
    
    def demote_to_archived(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """Archive a model version."""
        return self.transition_model_stage(
            model_name=model_name,
            version=version,
            stage="Archived",
            archive_existing=False
        )
    
    def get_latest_version(
        self,
        model_name: str,
        stage: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model, optionally filtered by stage.
        
        Args:
            model_name: Name of registered model
            stage: Optional stage filter
            
        Returns:
            Model version info dict or None
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(
                    name=model_name,
                    stages=[stage]
                )
            else:
                versions = self.client.get_latest_versions(
                    name=model_name
                )
            
            if not versions:
                logger.warning(f"No versions found for {model_name}")
                return None
            
            latest = versions[0]
            
            return {
                "name": latest.name,
                "version": latest.version,
                "stage": latest.current_stage,
                "status": latest.status,
                "run_id": latest.run_id,
                "source": latest.source,
                "description": latest.description,
                "creation_timestamp": latest.creation_timestamp
            }
            
        except MlflowException as e:
            logger.error(f"Failed to get latest version: {e}")
            return None
    
    def get_production_model(
        self,
        model_name: str
    ) -> Optional[Any]:
        """
        Load the production model for inference.
        
        Args:
            model_name: Name of registered model
            
        Returns:
            Loaded model object or None
        """
        logger.info(f"Loading production model: {model_name}")
        
        try:
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Successfully loaded production model")
            return model
            
        except MlflowException as e:
            logger.error(f"Failed to load production model: {e}")
            return None
    
    def get_staging_model(
        self,
        model_name: str
    ) -> Optional[Any]:
        """
        Load the staging model for validation.
        
        Args:
            model_name: Name of registered model
            
        Returns:
            Loaded model object or None
        """
        try:
            model_uri = f"models:/{model_name}/Staging"
            model = mlflow.sklearn.load_model(model_uri)
            return model
            
        except MlflowException as e:
            logger.error(f"Failed to load staging model: {e}")
            return None
    
    def list_model_versions(
        self,
        model_name: str
    ) -> List[Dict[str, Any]]:
        """
        List all versions of a registered model.
        
        Args:
            model_name: Name of registered model
            
        Returns:
            List of version info dicts
        """
        try:
            # Search for all versions
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "creation_timestamp": v.creation_timestamp,
                    "description": v.description
                }
                for v in versions
            ]
            
        except MlflowException as e:
            logger.error(f"Failed to list versions: {e}")
            return []
    
    def list_registered_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List of registered model info dicts
        """
        try:
            models = self.client.search_registered_models()
            
            return [
                {
                    "name": m.name,
                    "description": m.description,
                    "creation_timestamp": m.creation_timestamp,
                    "last_updated_timestamp": m.last_updated_timestamp,
                    "latest_versions": [
                        {"version": v.version, "stage": v.current_stage}
                        for v in (m.latest_versions or [])
                    ]
                }
                for m in models
            ]
            
        except MlflowException as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def delete_model_version(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """
        Delete a specific model version.
        
        Args:
            model_name: Name of registered model
            version: Version to delete
            
        Returns:
            True if successful
        """
        logger.warning(f"Deleting {model_name} version {version}")
        
        try:
            self.client.delete_model_version(
                name=model_name,
                version=version
            )
            logger.info(f"Deleted version {version}")
            return True
            
        except MlflowException as e:
            logger.error(f"Failed to delete version: {e}")
            return False
    
    def create_registered_model(
        self,
        model_name: str,
        description: str = None,
        tags: Dict[str, str] = None
    ) -> bool:
        """
        Create a new registered model entry.
        
        Args:
            model_name: Name for the new model
            description: Model description
            tags: Model tags
            
        Returns:
            True if successful
        """
        try:
            self.client.create_registered_model(
                name=model_name,
                description=description,
                tags=tags
            )
            logger.info(f"Created registered model: {model_name}")
            return True
            
        except MlflowException as e:
            if "already exists" in str(e).lower():
                logger.info(f"Model {model_name} already exists")
                return True
            logger.error(f"Failed to create model: {e}")
            return False
    
    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare metrics between two model versions.
        
        Args:
            model_name: Name of registered model
            version1: First version
            version2: Second version
            
        Returns:
            Comparison dict with metrics from both versions
        """
        comparison = {}
        
        for version in [version1, version2]:
            try:
                mv = self.client.get_model_version(
                    name=model_name,
                    version=version
                )
                
                run = self.client.get_run(mv.run_id)
                metrics = run.data.metrics
                
                comparison[f"v{version}"] = {
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "metrics": metrics
                }
                
            except MlflowException as e:
                logger.error(f"Failed to get version {version}: {e}")
                comparison[f"v{version}"] = {"error": str(e)}
        
        return comparison
    
    def auto_promote_best(
        self,
        model_name: str,
        metric: str = "test_rmse",
        lower_is_better: bool = True,
        min_improvement: float = 0.01
    ) -> Optional[str]:
        """
        Automatically promote the best model to production.
        
        Args:
            model_name: Name of registered model
            metric: Metric to compare
            lower_is_better: Whether lower metric is better
            min_improvement: Minimum improvement required to promote
            
        Returns:
            Promoted version number or None
        """
        logger.info(f"Auto-promoting best {model_name} based on {metric}")
        
        # Get current production model
        current_prod = self.get_latest_version(model_name, stage="Production")
        
        # Get all versions
        versions = self.list_model_versions(model_name)
        
        if not versions:
            logger.warning("No versions to compare")
            return None
        
        # Find best version
        best_version = None
        best_metric = float('inf') if lower_is_better else float('-inf')
        
        for v in versions:
            try:
                run = self.client.get_run(v["run_id"])
                metric_value = run.data.metrics.get(metric)
                
                if metric_value is None:
                    continue
                
                is_better = (
                    metric_value < best_metric if lower_is_better
                    else metric_value > best_metric
                )
                
                if is_better:
                    best_metric = metric_value
                    best_version = v["version"]
                    
            except Exception as e:
                logger.warning(f"Could not get metrics for v{v['version']}: {e}")
        
        if best_version is None:
            logger.warning(f"No valid versions found with metric {metric}")
            return None
        
        # Check if improvement is significant
        if current_prod:
            try:
                current_run = self.client.get_run(current_prod["run_id"])
                current_metric = current_run.data.metrics.get(metric)
                
                if current_metric:
                    improvement = abs(current_metric - best_metric) / abs(current_metric)
                    
                    if improvement < min_improvement:
                        logger.info(
                            f"Improvement {improvement:.2%} below threshold {min_improvement:.2%}"
                        )
                        return None
                        
            except Exception as e:
                logger.warning(f"Could not compare with current: {e}")
        
        # Promote best version
        if self.promote_to_production(model_name, best_version):
            logger.info(f"Promoted v{best_version} to Production ({metric}: {best_metric:.4f})")
            return best_version
        
        return None


def register_best_model(
    run_id: str,
    model_name: str = "crypto-price-predictor",
    promote_to: str = "Staging"
) -> Dict[str, Any]:
    """
    Register and optionally promote a model.
    
    Args:
        run_id: MLflow run ID
        model_name: Name for registered model
        promote_to: Stage to promote to (None, Staging, Production)
        
    Returns:
        Registration result dict
    """
    registry = ModelRegistry()
    
    # Register the model
    version = registry.register_model(
        run_id=run_id,
        model_name=model_name,
        description=f"Bitcoin price prediction model from run {run_id}",
        tags={
            "registered_at": datetime.now().isoformat(),
            "dataset": "bitcoin",
            "target": "price_1h"
        }
    )
    
    result = {
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "stage": "None"
    }
    
    # Promote if requested
    if promote_to and promote_to in ["Staging", "Production"]:
        if promote_to == "Staging":
            registry.promote_to_staging(model_name, version)
        elif promote_to == "Production":
            registry.promote_to_production(model_name, version)
        
        result["stage"] = promote_to
    
    logger.info(f"Model registered: {model_name} v{version} ({promote_to or 'None'})")
    
    return result


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    
    print("\nRegistered Models:")
    print("-" * 40)
    for model in registry.list_registered_models():
        print(f"  {model['name']}")
        for v in model.get('latest_versions', []):
            print(f"    - v{v['version']} ({v['stage']})")
