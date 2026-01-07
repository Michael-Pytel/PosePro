import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

from .model_loader import get_feature_names

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts features from Django metrics dict for ML prediction.
    Ensures features are in correct order and handles missing values.
    """
    
    # Feature mapping: Django key path -> ML feature name
    FEATURE_MAPPINGS = {
        'range_of_motion': {
            'max_elbow_angle': ('range_of_motion', 'max_elbow_angle'),
            'min_elbow_angle': ('range_of_motion', 'min_elbow_angle'),
            'range_of_motion_angle': ('range_of_motion', 'range_of_motion'),
            'full_depth': ('range_of_motion', 'full_depth'),
            'full_lockout': ('range_of_motion', 'full_lockout'),
            'is_complete_rep': ('range_of_motion', 'is_complete_rep'),
            'down_time': ('timing', 'down_time'),
            'up_time': ('timing', 'up_time'),
            'rep_time': ('timing', 'rep_time'),
            'bottom_pause': ('timing', 'bottom_pause')
        },
        'hips': {
            'hip_deviation_bottom': ('plank', 'hip_deviation_bottom'),
            'hip_deviation_max_sag': ('plank', 'hip_deviation_max_sag'),
            'hip_deviation_max_pike': ('plank', 'hip_deviation_max_pike'),
            'hip_deviation_mean_abs': ('plank', 'hip_deviation_mean_abs'),
            'torso_angle_bottom_deg': ('plank', 'torso_angle_bottom_deg'),
            'torso_angle_max_deg': ('plank', 'torso_angle_max_deg'),
            'torso_angle_min_deg': ('plank', 'torso_angle_min_deg'),
            'torso_angle_mean_deg': ('plank', 'torso_angle_mean_deg'),
            'torso_angle_range_deg': ('plank', 'torso_angle_range_deg'),
            'body_angle_bottom_deg': ('plank', 'body_angle_bottom_deg'),
            'body_angle_max_deg': ('plank', 'body_angle_max_deg'),
            'body_angle_min_deg': ('plank', 'body_angle_min_deg'),
            'body_angle_mean_deg': ('plank', 'body_angle_mean_deg'),
            'body_angle_range_deg': ('plank', 'body_angle_range_deg')
        },
        'head_position': {
            'torso_head_angle_bottom': ('head', 'torso_head_angle_bottom'),
            'torso_head_angle_min': ('head', 'torso_head_angle_min'),
            'torso_head_angle_max': ('head', 'torso_head_angle_max'),
            'torso_head_angle_range': ('head', 'torso_head_angle_range'),
            'torso_head_angle_mean': ('head', 'torso_head_angle_mean'),
            'head_forward_norm_bottom': ('head', 'head_forward_norm_bottom'),
            'head_forward_norm_min': ('head', 'head_forward_norm_min'),
            'head_forward_norm_max': ('head', 'head_forward_norm_max'),
            'head_forward_norm_range': ('head', 'head_forward_norm_range'),
            'head_forward_norm_mean': ('head', 'head_forward_norm_mean')
        }
    }
    
    def __init__(self):
        """Initialize the feature extractor"""
        pass
    
    def _get_nested_value(self, metrics_dict: Dict, path: tuple) -> Optional[float]:
        """
        Safely get a nested value from metrics dict.
        
        Args:
            metrics_dict: The metrics dictionary from Django
            path: Tuple of (section, key) e.g., ('plank', 'hip_deviation_bottom')
            
        Returns:
            The value or None if not found
        """
        section, key = path
        
        try:
            section_data = metrics_dict.get(section)
            if section_data is None:
                return None
            
            value = section_data.get(key)
            
            # Convert boolean to int
            if isinstance(value, bool):
                return int(value)
            
            return value
            
        except Exception as e:
            logger.warning(f"Error getting {path}: {e}")
            return None
    
    def extract_features(self, metrics_dict: Dict, criterion: str) -> Optional[np.ndarray]:
        """
        Extract features for a specific criterion.
        
        Args:
            metrics_dict: Dictionary from aggregate_repetition_metrics() containing:
                {
                    'timing': {...},
                    'range_of_motion': {...},
                    'plank': {...},
                    'head': {...}
                }
            criterion: One of 'range_of_motion', 'hips', 'head_position'
            
        Returns:
            numpy array of features in correct order, or None if extraction fails
        """
        if criterion not in self.FEATURE_MAPPINGS:
            logger.error(f"Unknown criterion: {criterion}")
            return None
        
        # Get expected feature names from model
        try:
            expected_features = get_feature_names(criterion)
        except Exception as e:
            logger.error(f"Failed to get feature names for {criterion}: {e}")
            return None
        
        # Extract features in correct order
        feature_mapping = self.FEATURE_MAPPINGS[criterion]
        feature_values = []
        missing_features = []
        
        for feature_name in expected_features:
            if feature_name not in feature_mapping:
                logger.error(f"Feature {feature_name} not in mapping for {criterion}")
                return None
            
            path = feature_mapping[feature_name]
            value = self._get_nested_value(metrics_dict, path)
            
            if value is None:
                missing_features.append(feature_name)
                # Use NaN for missing values - model's scaler can handle it
                value = np.nan
            
            feature_values.append(value)
        
        if missing_features:
            logger.warning(
                f"{criterion}: Missing features {missing_features}. "
                f"Predictions may be unreliable."
            )
        
        # Convert to numpy array
        features = pd.DataFrame(
            [feature_values], 
            columns=expected_features
        )   
        
        logger.debug(f"Extracted {len(feature_values)} features for {criterion}")
        
        return features
    
    def extract_all_features(self, metrics_dict: Dict) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract features for all three criteria.
        
        Args:
            metrics_dict: Complete metrics dictionary
            
        Returns:
            Dictionary mapping criterion -> feature array
        """
        results = {}
        
        for criterion in ['range_of_motion', 'hips', 'head_position']:
            try:
                features = self.extract_features(metrics_dict, criterion)
                results[criterion] = features
                
                if features is not None:
                    logger.info(f"✓ Extracted {criterion} features: shape {features.shape}")
                else:
                    logger.warning(f"✗ Failed to extract {criterion} features")
                    
            except Exception as e:
                logger.error(f"Error extracting {criterion} features: {e}")
                results[criterion] = None
        
        return results
    
    def validate_features(self, features: np.ndarray, criterion: str) -> bool:
        """
        Validate that extracted features are reasonable.
        
        Args:
            features: Feature array
            criterion: Criterion name
            
        Returns:
            True if features look valid
        """
        if features is None:
            return False
        
        # Check shape
        expected_features = get_feature_names(criterion)
        expected_count = len(expected_features)
        
        if features.shape != (1, expected_count):
            logger.error(
                f"{criterion}: Shape mismatch. "
                f"Expected (1, {expected_count}), got {features.shape}"
            )
            return False
        
        # Check for all NaN
        if np.all(np.isnan(features)):
            logger.error(f"{criterion}: All features are NaN")
            return False
        
        # Check for inf
        if np.any(np.isinf(features)):
            logger.warning(f"{criterion}: Contains infinite values")
            return False
        
        return True
    
    def features_to_dataframe(self, features: np.ndarray, criterion: str) -> pd.DataFrame:
        """
        Convert feature array to DataFrame with column names.
        Useful for debugging.
        
        Args:
            features: Feature array
            criterion: Criterion name
            
        Returns:
            DataFrame with named columns
        """
        feature_names = get_feature_names(criterion)
        return pd.DataFrame(features, columns=feature_names)


# Convenience functions
_extractor = None

def get_extractor() -> FeatureExtractor:
    """Get singleton FeatureExtractor instance"""
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


def extract_features(metrics_dict: Dict, criterion: str) -> Optional[np.ndarray]:
    """Convenience function to extract features"""
    return get_extractor().extract_features(metrics_dict, criterion)


def extract_all_features(metrics_dict: Dict) -> Dict[str, Optional[np.ndarray]]:
    """Convenience function to extract all features"""
    return get_extractor().extract_all_features(metrics_dict)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example metrics dict (structure from Django)
    example_metrics = {
        'timing': {
            'down_time': 0.8,
            'up_time': 0.6,
            'rep_time': 1.4,
            'bottom_pause': 0.1
        },
        'range_of_motion': {
            'max_elbow_angle': 175.0,
            'min_elbow_angle': 85.0,
            'range_of_motion': 90.0,
            'full_depth': True,
            'full_lockout': True,
            'is_complete_rep': True
        },
        'plank': {
            'hip_deviation_bottom': 5.2,
            'hip_deviation_max_sag': 8.1,
            'hip_deviation_max_pike': -3.2,
            'hip_deviation_mean_abs': 4.5,
            'torso_angle_bottom_deg': 165.0,
            'torso_angle_max_deg': 170.0,
            'torso_angle_min_deg': 160.0,
            'torso_angle_mean_deg': 165.5,
            'torso_angle_range_deg': 10.0,
            'body_angle_bottom_deg': 175.0,
            'body_angle_max_deg': 178.0,
            'body_angle_min_deg': 172.0,
            'body_angle_mean_deg': 175.0,
            'body_angle_range_deg': 6.0
        },
        'head': {
            'torso_head_angle_bottom': 165.0,
            'torso_head_angle_min': 160.0,
            'torso_head_angle_max': 170.0,
            'torso_head_angle_range': 10.0,
            'torso_head_angle_mean': 165.0,
            'head_forward_norm_bottom': 0.05,
            'head_forward_norm_min': 0.02,
            'head_forward_norm_max': 0.08,
            'head_forward_norm_range': 0.06,
            'head_forward_norm_mean': 0.05
        }
    }
    
    print("\n=== Testing Feature Extraction ===\n")
    
    extractor = FeatureExtractor()
    
    # Test each criterion
    for criterion in ['range_of_motion', 'hips', 'head_position']:
        print(f"\n{criterion.upper()}:")
        
        features = extractor.extract_features(example_metrics, criterion)
        
        if features is not None:
            print(f"  Shape: {features.shape}")
            print(f"  Valid: {extractor.validate_features(features, criterion)}")
            
            # Show as DataFrame
            df = extractor.features_to_dataframe(features, criterion)
            print(f"\n  Features:")
            print(df.T)  # Transpose for better display
        else:
            print("  Failed to extract features")