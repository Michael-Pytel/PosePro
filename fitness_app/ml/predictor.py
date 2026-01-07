"""
ML Predictor
Makes predictions on exercise form using trained ML models.
"""

import numpy as np
from typing import Dict, Optional, Any
import logging

from .model_loader import get_loader
from .feature_extractor import get_extractor

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Predicts exercise form correctness using trained ML models.
    Combines model loading, feature extraction, and prediction logic.
    """
    
    # Confidence level thresholds
    CONFIDENCE_THRESHOLDS = {
        'high': 0.8,      # prob > 0.8 or < 0.2
        'medium': 0.65,   # prob > 0.65 or < 0.35
        'low': 0.5        # Everything else
    }
    
    def __init__(self):
        """Initialize predictor with model loader and feature extractor"""
        self.model_loader = get_loader()
        self.feature_extractor = get_extractor()
        
    def _get_confidence_level(self, probability: float) -> str:
        """
        Determine confidence level based on probability.
        
        Args:
            probability: Predicted probability (0-1)
            
        Returns:
            'high', 'medium', or 'low'
        """
        # Distance from decision boundary (0.5)
        distance = abs(probability - 0.5)
        
        if distance >= (self.CONFIDENCE_THRESHOLDS['high'] - 0.5):
            return 'high'
        elif distance >= (self.CONFIDENCE_THRESHOLDS['medium'] - 0.5):
            return 'medium'
        else:
            return 'low'
    
    def predict_single_criterion(
        self, 
        metrics_dict: Dict, 
        criterion: str
    ) -> Optional[Dict[str, Any]]:
        """
        Make prediction for a single criterion.
        
        Args:
            metrics_dict: Dictionary from aggregate_repetition_metrics()
            criterion: One of 'range_of_motion', 'hips', 'head_position'
            
        Returns:
            Dictionary with prediction results:
            {
                'criterion': str,
                'prediction': int (0=correct, 1=incorrect),
                'probability': float (0-1),
                'confidence': str ('high', 'medium', 'low'),
                'threshold_used': float,
                'success': bool,
                'error': str (if failed)
            }
            Returns None if prediction fails completely
        """
        try:
            # 1. Extract features
            features = self.feature_extractor.extract_features(metrics_dict, criterion)
            
            if features is None:
                logger.error(f"{criterion}: Feature extraction failed")
                return {
                    'criterion': criterion,
                    'success': False,
                    'error': 'Feature extraction failed'
                }
            
            # 2. Validate features
            if not self.feature_extractor.validate_features(features, criterion):
                logger.error(f"{criterion}: Feature validation failed")
                return {
                    'criterion': criterion,
                    'success': False,
                    'error': 'Invalid features'
                }
            
            # 3. Load model and threshold
            model_obj = self.model_loader.load_model(criterion)
            pipeline = model_obj['pipeline']
            threshold = model_obj['threshold']
            
            # 4. Get probability from pipeline
            # Pipeline returns probabilities for both classes [prob_class_0, prob_class_1]
            probabilities = pipeline.predict_proba(features)[0]
            probability_incorrect = float(probabilities[1])  # Probability of class 1 (incorrect)
            
            # 5. Apply custom threshold
            prediction = 1 if probability_incorrect >= threshold else 0
            
            # 6. Determine confidence
            confidence = self._get_confidence_level(probability_incorrect)
            
            # 7. Return results
            result = {
                'criterion': criterion,
                'prediction': prediction,
                'probability': probability_incorrect,
                'confidence': confidence,
                'threshold_used': threshold,
                'success': True
            }
            
            logger.info(
                f"{criterion}: prediction={prediction}, "
                f"prob={probability_incorrect:.3f}, "
                f"confidence={confidence}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {criterion}: {e}", exc_info=True)
            return {
                'criterion': criterion,
                'success': False,
                'error': str(e)
            }
    
    def predict_all_criteria(self, metrics_dict: Dict) -> Dict[str, Dict[str, Any]]:
        """
        Make predictions for all three criteria.
        
        Args:
            metrics_dict: Complete metrics dictionary from Django
            
        Returns:
            Dictionary mapping criterion -> prediction results
        """
        results = {}
        
        for criterion in ['range_of_motion', 'hips', 'head_position']:
            try:
                prediction = self.predict_single_criterion(metrics_dict, criterion)
                results[criterion] = prediction
                
            except Exception as e:
                logger.error(f"Error predicting {criterion}: {e}")
                results[criterion] = {
                    'criterion': criterion,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def format_prediction_for_display(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format prediction for frontend display.
        
        Args:
            prediction: Raw prediction dict
            
        Returns:
            Formatted dict suitable for display:
            {
                'status': bool,           # True = correct, False = incorrect
                'label': str,             # Display label
                'value': str,             # 'Correct' or 'Incorrect'
                'confidence': str,        # 'High', 'Medium', 'Low'
                'probability': float,     # 0-1
                'ai_powered': bool        # True
            }
        """
        if not prediction.get('success', False):
            return {
                'status': None,
                'label': prediction['criterion'].replace('_', ' ').title(),
                'value': 'Error',
                'confidence': 'N/A',
                'probability': None,
                'ai_powered': True,
                'error': prediction.get('error', 'Unknown error')
            }
        
        is_correct = prediction['prediction'] == 0
        
        return {
            'status': is_correct,
            'label': prediction['criterion'].replace('_', ' ').title() + ' (AI)',
            'value': 'Correct' if is_correct else 'Incorrect',
            'confidence': prediction['confidence'].title(),
            'probability': round(prediction['probability'], 3),
            'ai_powered': True
        }
    
    def get_overall_assessment(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Get overall form assessment based on all criteria.
        
        Args:
            predictions: Dictionary of all predictions
            
        Returns:
            Overall assessment summary
        """
        successful_predictions = [
            p for p in predictions.values() 
            if p.get('success', False)
        ]
        
        if not successful_predictions:
            return {
                'overall_correct': False,
                'criteria_passed': 0,
                'criteria_total': 0,
                'confidence': 'N/A',
                'message': 'Unable to assess form'
            }
        
        # Count correct predictions
        correct_count = sum(
            1 for p in successful_predictions 
            if p['prediction'] == 0
        )
        total_count = len(successful_predictions)
        
        # Overall is correct only if ALL criteria are correct
        overall_correct = correct_count == total_count
        
        # Average confidence (simplified: high=3, medium=2, low=1)
        confidence_scores = {
            'high': 3,
            'medium': 2,
            'low': 1
        }
        avg_confidence_score = np.mean([
            confidence_scores.get(p.get('confidence', 'low'), 1)
            for p in successful_predictions
        ])
        
        if avg_confidence_score >= 2.5:
            overall_confidence = 'high'
        elif avg_confidence_score >= 1.5:
            overall_confidence = 'medium'
        else:
            overall_confidence = 'low'
        
        return {
            'overall_correct': overall_correct,
            'criteria_passed': correct_count,
            'criteria_total': total_count,
            'confidence': overall_confidence,
            'message': self._get_feedback_message(correct_count, total_count)
        }
    
    def _get_feedback_message(self, correct: int, total: int) -> str:
        """Generate user-friendly feedback message"""
        if correct == total:
            return "Excellent form! All criteria met."
        elif correct == 0:
            return "Form needs improvement across all criteria."
        else:
            return f"Good effort! {correct} of {total} criteria met. Review the highlighted areas."


# Convenience functions
_predictor = None

def get_predictor() -> MLPredictor:
    """Get singleton MLPredictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor


def predict_single_criterion(metrics_dict: Dict, criterion: str) -> Optional[Dict]:
    """Convenience function to predict single criterion"""
    return get_predictor().predict_single_criterion(metrics_dict, criterion)


def predict_all_criteria(metrics_dict: Dict) -> Dict[str, Dict]:
    """Convenience function to predict all criteria"""
    return get_predictor().predict_all_criteria(metrics_dict)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example metrics dict (same structure as Django)
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
    
    print("\n=== Testing ML Predictor ===\n")
    
    predictor = MLPredictor()
    
    # Test all criteria
    predictions = predictor.predict_all_criteria(example_metrics)
    
    print("\n=== Predictions ===\n")
    for criterion, pred in predictions.items():
        if pred['success']:
            formatted = predictor.format_prediction_for_display(pred)
            print(f"{formatted['label']}:")
            print(f"  Result: {formatted['value']}")
            print(f"  Confidence: {formatted['confidence']}")
            print(f"  Probability (incorrect): {formatted['probability']:.3f}")
        else:
            print(f"{criterion}: FAILED - {pred.get('error')}")
        print()
    
    # Overall assessment
    overall = predictor.get_overall_assessment(predictions)
    print("\n=== Overall Assessment ===")
    print(f"Status: {'✓' if overall['overall_correct'] else '✗'}")
    print(f"Criteria: {overall['criteria_passed']}/{overall['criteria_total']}")
    print(f"Confidence: {overall['confidence']}")
    print(f"Message: {overall['message']}")