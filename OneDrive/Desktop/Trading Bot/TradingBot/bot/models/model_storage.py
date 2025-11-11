"""Model storage and versioning for trained ML models."""

import json
import joblib
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelStorage:
    """Manages storage, versioning, and metadata for trained models."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize model storage.
        
        Args:
            storage_dir: Directory for storing models (default: TradingBot/models/)
        """
        if storage_dir is None:
            from bot.config import get_config
            config = get_config()
            # Use models directory at project root
            storage_dir = Path(__file__).parent.parent.parent / "models"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Metadata directory
        self.metadata_dir = self.storage_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
    
    def _get_model_path(self, name: str, version: Optional[str] = None) -> Path:
        """Get file path for a model.
        
        Args:
            name: Model name
            version: Model version (if None, uses latest)
        
        Returns:
            Path to model file
        """
        if version is None:
            version = self.get_latest_version(name)
            if version is None:
                raise ValueError(f"No versions found for model: {name}")
        
        return self.storage_dir / f"{name}_v{version}.pkl"
    
    def _get_metadata_path(self, name: str, version: Optional[str] = None) -> Path:
        """Get metadata file path for a model.
        
        Args:
            name: Model name
            version: Model version (if None, uses latest)
        
        Returns:
            Path to metadata file
        """
        if version is None:
            version = self.get_latest_version(name)
            if version is None:
                raise ValueError(f"No versions found for model: {name}")
        
        return self.metadata_dir / f"{name}_v{version}.json"
    
    def save_model(
        self,
        model: Any,
        name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a trained model with versioning.
        
        Args:
            model: Trained model object (must be pickle-able)
            name: Model name (e.g., "gmm_regime", "ensemble_predictor")
            version: Version string (if None, auto-increments)
            metadata: Additional metadata (training date, metrics, features, etc.)
        
        Returns:
            Version string of saved model
        """
        # Auto-increment version if not provided
        if version is None:
            latest_version = self.get_latest_version(name)
            if latest_version is None:
                version = "1.0.0"
            else:
                # Increment patch version
                parts = latest_version.split(".")
                if len(parts) == 3:
                    major, minor, patch = parts
                    version = f"{major}.{minor}.{int(patch) + 1}"
                else:
                    version = f"{latest_version}.1"
        
        # Prepare metadata
        model_metadata = {
            "name": name,
            "version": version,
            "saved_at": datetime.now().isoformat(),
            "model_type": type(model).__name__,
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        # Save model
        model_path = self._get_model_path(name, version)
        joblib.dump(model, model_path)
        logger.info(f"Saved model {name} v{version} to {model_path}")
        
        # Save metadata
        metadata_path = self._get_metadata_path(name, version)
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        return version
    
    def load_model(self, name: str, version: Optional[str] = None) -> Any:
        """Load a trained model.
        
        Args:
            name: Model name
            version: Model version (if None, loads latest)
        
        Returns:
            Loaded model object
        """
        model_path = self._get_model_path(name, version)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded model {name} v{version or 'latest'} from {model_path}")
        return model
    
    def load_metadata(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Load model metadata.
        
        Args:
            name: Model name
            version: Model version (if None, loads latest)
        
        Returns:
            Metadata dictionary
        """
        metadata_path = self._get_metadata_path(name, version)
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def list_models(self) -> List[str]:
        """List all available model names.
        
        Returns:
            List of model names
        """
        models = set()
        for model_file in self.storage_dir.glob("*.pkl"):
            # Extract name from filename (e.g., "gmm_regime_v1.0.0.pkl" -> "gmm_regime")
            name = model_file.stem.rsplit("_v", 1)[0]
            models.add(name)
        return sorted(list(models))
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions for a model.
        
        Args:
            name: Model name
        
        Returns:
            List of version strings, sorted (newest first)
        """
        versions = []
        for model_file in self.storage_dir.glob(f"{name}_v*.pkl"):
            # Extract version from filename
            version = model_file.stem.rsplit("_v", 1)[1]
            versions.append(version)
        
        # Sort versions (simple string sort, may need improvement for semantic versioning)
        return sorted(versions, reverse=True)
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version for a model.
        
        Args:
            name: Model name
        
        Returns:
            Latest version string, or None if no versions exist
        """
        versions = self.list_versions(name)
        return versions[0] if versions else None
    
    def delete_model(self, name: str, version: str) -> None:
        """Delete a specific model version.
        
        Args:
            name: Model name
            version: Model version
        """
        model_path = self._get_model_path(name, version)
        metadata_path = self._get_metadata_path(name, version)
        
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Deleted model {name} v{version}")
        
        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"Deleted metadata for {name} v{version}")

