import os
import json
import joblib
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton class for loading and caching ML models.
    Models are loaded once on first access and cached in memory.
    """
    
    _instance = None
    _models_cache = {}
    _metadata_cache = {}
    _initialized = False
    
    # Criterion name mapping (internal name -> file name)
    CRITERION_MAP = {
        'range_of_motion': 'rom',
        'hips': 'hips',
        'head_position': 'head'
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.models_dir = self._get_models_directory()
            self._initialized = True
            logger.info(f"ModelLoader initialized with models_dir: {self.models_dir}")
    
    def _get_models_directory(self) -> Path:
        """Get the absolute path to the models directory"""
        current_file = Path(__file__).resolve()
        models_dir = current_file.parent / 'models'
        
        if not models_dir.exists():
            logger.error(f"Models directory not found: {models_dir}")
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        return models_dir
    
    def _get_file_paths(self, criterion: str) -> tuple[Path, Path]:
        """
        Get pipeline and metadata file paths for a criterion.
        
        Args:
            criterion: One of 'range_of_motion', 'hips', 'head_position'
            
        Returns:
            Tuple of (pipeline_path, metadata_path)
        """
        if criterion not in self.CRITERION_MAP:
            raise ValueError(
                f"Unknown criterion: {criterion}. "
                f"Must be one of {list(self.CRITERION_MAP.keys())}"
            )
        
        file_prefix = self.CRITERION_MAP[criterion]
        pipeline_path = self.models_dir / f"{file_prefix}_pipeline_latest.pkl"
        metadata_path = self.models_dir / f"{file_prefix}_metadata_latest.json"
        
        return pipeline_path, metadata_path
    
    def _load_pipeline(self, pipeline_path: Path) -> Any:
        """Load a pickled sklearn/imblearn pipeline"""
        try:
            pipeline = joblib.load(pipeline_path)
            logger.info(f"Loaded pipeline from {pipeline_path}")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load pipeline from {pipeline_path}: {e}")
            raise
    
    def _load_metadata(self, metadata_path: Path) -> Dict:
        """Load model metadata JSON file"""
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            raise
    
    def load_model(self, criterion: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load model pipeline and metadata for a criterion.
        
        Args:
            criterion: One of 'range_of_motion', 'hips', 'head_position'
            force_reload: If True, reload even if cached
            
        Returns:
            Dictionary containing:
                - pipeline: The trained sklearn/imblearn pipeline
                - threshold: Optimal decision threshold
                - feature_names: List of expected feature names
                - metadata: Full metadata dict
        """
        # Check cache first
        if not force_reload and criterion in self._models_cache:
            logger.debug(f"Using cached model for {criterion}")
            return self._models_cache[criterion]
        
        # Load from disk
        pipeline_path, metadata_path = self._get_file_paths(criterion)
        
        # Verify files exist
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load pipeline and metadata
        pipeline = self._load_pipeline(pipeline_path)
        metadata = self._load_metadata(metadata_path)
        
        # Create model object
        model_obj = {
            'pipeline': pipeline,
            'threshold': metadata['threshold'],
            'feature_names': metadata['feature_names'],
            'metadata': metadata,
            'criterion': criterion
        }
        
        # Cache it
        self._models_cache[criterion] = model_obj
        logger.info(f"Cached model for {criterion}")
        
        return model_obj
    
    def get_pipeline(self, criterion: str):
        """Get just the pipeline for a criterion"""
        model = self.load_model(criterion)
        return model['pipeline']
    
    def get_threshold(self, criterion: str) -> float:
        """Get the optimal threshold for a criterion"""
        model = self.load_model(criterion)
        return model['threshold']
    
    def get_feature_names(self, criterion: str) -> list:
        """Get the expected feature names for a criterion"""
        model = self.load_model(criterion)
        return model['feature_names']
    
    def load_all_models(self) -> Dict[str, Dict]:
        """
        Load all models at once (useful for startup).
        
        Returns:
            Dictionary mapping criterion -> model object
        """
        all_models = {}
        
        for criterion in self.CRITERION_MAP.keys():
            try:
                all_models[criterion] = self.load_model(criterion)
                logger.info(f"✓ Loaded {criterion} model")
            except Exception as e:
                logger.error(f"✗ Failed to load {criterion} model: {e}")
                all_models[criterion] = None
        
        return all_models
    
    def is_model_loaded(self, criterion: str) -> bool:
        """Check if a model is already loaded in cache"""
        return criterion in self._models_cache
    
    def clear_cache(self):
        """Clear all cached models (useful for reloading)"""
        self._models_cache.clear()
        self._metadata_cache.clear()
        logger.info("Cleared model cache")
    
    def get_model_info(self, criterion: str) -> Dict:
        """
        Get information about a model without loading the pipeline.
        
        Returns:
            Dictionary with model information from metadata
        """
        _, metadata_path = self._get_file_paths(criterion)
        metadata = self._load_metadata(metadata_path)
        
        return {
            'criterion': criterion,
            'model_type': metadata['model_type'],
            'threshold': metadata['threshold'],
            'feature_count': len(metadata['feature_names']),
            'features': metadata['feature_names'],
            'metrics': metadata['metrics'],
            'timestamp': metadata['timestamp']
        }


# Convenience functions for easy access
_loader = None

def get_loader() -> ModelLoader:
    """Get the singleton ModelLoader instance"""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader


def load_model(criterion: str) -> Dict[str, Any]:
    """Convenience function to load a model"""
    return get_loader().load_model(criterion)


def get_pipeline(criterion: str):
    """Convenience function to get a pipeline"""
    return get_loader().get_pipeline(criterion)


def get_threshold(criterion: str) -> float:
    """Convenience function to get a threshold"""
    return get_loader().get_threshold(criterion)


def get_feature_names(criterion: str) -> list:
    """Convenience function to get feature names"""
    return get_loader().get_feature_names(criterion)

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Test loading
    loader = ModelLoader()
    
    print("\n=== Testing Model Loader ===\n")
    
    # Load all models
    models = loader.load_all_models()
    
    print("\n=== Model Information ===\n")
    for criterion in ['range_of_motion', 'hips', 'head_position']:
        try:
            info = loader.get_model_info(criterion)
            print(f"{criterion.upper()}:")
            print(f"  Model: {info['model_type']}")
            print(f"  Threshold: {info['threshold']:.4f}")
            print(f"  Features: {info['feature_count']}")
            print(f"  F1-Score: {info['metrics']['F1-Score']:.3f}")
            print(f"  Recall: {info['metrics']['Recall']:.3f}")
            print()
        except Exception as e:
            print(f"{criterion}: ERROR - {e}\n")
    
    # Test individual loading
    print("\n=== Testing Individual Access ===\n")
    try:
        rom_pipeline = get_pipeline('range_of_motion')
        rom_threshold = get_threshold('range_of_motion')
        rom_features = get_feature_names('range_of_motion')
        
        print(f"ROM Pipeline: {type(rom_pipeline)}")
        print(f"ROM Threshold: {rom_threshold:.4f}")
        print(f"ROM Features: {rom_features}")
    except Exception as e:
        print(f"ERROR: {e}")