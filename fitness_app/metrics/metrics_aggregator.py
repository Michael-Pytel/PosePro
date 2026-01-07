from typing import Dict, List, Any
import logging

from fitness_app.metrics.report_repetition_tools_basic import get_timing, get_rom
from fitness_app.metrics.report_repetition_tools_plank import get_plank
from fitness_app.metrics.report_repetition_tools_head import get_head
from fitness_app.ml.predictor import get_predictor  # NEW IMPORT

logger = logging.getLogger(__name__)


def aggregate_repetition_metrics(signals, visibility_scores, repetition, fps) -> Dict[str, Any]:
    """
    Compute all metrics for a single repetition and make ML predictions.
    
    Returns:
        Dictionary containing:
        - timing: Timing metrics (down_time, up_time, etc.)
        - range_of_motion: ROM metrics (angles, depth, lockout)
        - plank: Hip/body alignment metrics
        - head: Head position metrics
        - ml_predictions: ML model predictions for each criterion (NEW)
        - frame_info: Frame numbers
    """
    
    # === Compute raw metrics (existing code) ===
    try:
        timing = get_timing(signals, visibility_scores, repetition, fps)
    except Exception as e:
        logger.error(f"Error getting timing metrics: {e}")
        timing = None
    
    try:
        rom = get_rom(signals, visibility_scores, repetition, fps)
    except Exception as e:
        logger.error(f"Error getting ROM metrics: {e}")
        rom = None
    
    try:
        plank = get_plank(signals, visibility_scores, repetition, fps)
    except Exception as e:
        logger.error(f"Error getting plank metrics: {e}")
        plank = None
    
    try:
        head = get_head(signals, visibility_scores, repetition)
    except Exception as e:
        logger.error(f"Error getting head metrics: {e}")
        head = None
    
    # === Build metrics dict ===
    metrics_dict = {
        'timing': timing,
        'range_of_motion': rom,
        'plank': plank,
        'head': head,
        'frame_info': {
            'start_frame': repetition['start_frame'],
            'end_frame': repetition['end_frame'],
            'bottom_frame': repetition.get('bottom_frame')
        }
    }
    
    # === NEW: Make ML predictions ===
    ml_predictions = None
    try:
        predictor = get_predictor()
        ml_predictions = predictor.predict_all_criteria(metrics_dict)
        logger.info("✓ ML predictions completed")
    except Exception as e:
        logger.error(f"ML prediction failed: {e}", exc_info=True)
        ml_predictions = {
            'range_of_motion': {'success': False, 'error': str(e)},
            'hips': {'success': False, 'error': str(e)},
            'head_position': {'success': False, 'error': str(e)}
        }
    
    # Add ML predictions to output
    metrics_dict['ml_predictions'] = ml_predictions
    
    return metrics_dict


def format_metrics_for_display(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metrics for frontend display.
    Now includes both rule-based checks AND ML predictions.
    
    Returns:
        Dictionary with:
        - summary: Quick overview metrics
        - form_checks: Rule-based form checks (existing)
        - ml_form_checks: ML-powered form checks (NEW)
        - detailed: Full raw metrics
    """
    
    formatted = {
        'summary': {},
        'detailed': {},
        'form_checks': {},      # Rule-based (existing)
        'ml_form_checks': {}    # ML predictions (NEW)
    }
    
    # === Summary Metrics (existing code) ===
    if metrics.get('timing'):
        formatted['summary']['total_time'] = f"{metrics['timing']['rep_time']:.2f}s"
        formatted['summary']['down_time'] = f"{metrics['timing']['down_time']:.2f}s"
        formatted['summary']['up_time'] = f"{metrics['timing']['up_time']:.2f}s"
    
    if metrics.get('range_of_motion'):
        formatted['summary']['rom'] = f"{metrics['range_of_motion']['range_of_motion']:.1f}°"
        formatted['summary']['min_angle'] = f"{metrics['range_of_motion']['min_elbow_angle']:.1f}°"
        formatted['summary']['max_angle'] = f"{metrics['range_of_motion']['max_elbow_angle']:.1f}°"
    
    # === Rule-based Form Checks (existing code) ===
    if metrics.get('range_of_motion'):
        formatted['form_checks']['full_depth'] = {
            'status': metrics['range_of_motion']['full_depth'],
            'label': 'Full Depth (Rule)',
            'value': 'Correct' if metrics['range_of_motion']['full_depth'] else 'Incorrect'
        }
        formatted['form_checks']['full_lockout'] = {
            'status': metrics['range_of_motion']['full_lockout'],
            'label': 'Full Lockout (Rule)',
            'value': 'Correct' if metrics['range_of_motion']['full_lockout'] else 'Incorrect'
        }
        formatted['form_checks']['complete_rep'] = {
            'status': metrics['range_of_motion']['is_complete_rep'],
            'label': 'Complete Rep (Rule)',
            'value': 'Yes' if metrics['range_of_motion']['is_complete_rep'] else 'No'
        }
    
    if metrics.get('plank'):
        hip_deviation = metrics['plank']['hip_deviation_mean_abs']
        formatted['form_checks']['hip_alignment'] = {
            'status': hip_deviation < 10,
            'label': 'Hip Alignment (Rule)',
            'value': f"{hip_deviation:.1f}cm deviation"
        }
        
        body_angle = metrics['plank']['body_angle_mean_deg']
        formatted['form_checks']['body_straight'] = {
            'status': 160 <= body_angle <= 200,
            'label': 'Body Position (Rule)',
            'value': 'Straight' if 160 <= body_angle <= 200 else 'Not Straight'
        }
    
    if metrics.get('head'):
        head_angle = metrics['head']['torso_head_angle_mean']
        formatted['form_checks']['head_position'] = {
            'status': 150 <= head_angle <= 190,
            'label': 'Head Position (Rule)',
            'value': 'Neutral' if 150 <= head_angle <= 190 else 'Not Neutral'
        }
    
    # === NEW: ML-powered Form Checks ===
    if metrics.get('ml_predictions'):
        predictor = get_predictor()
        
        for criterion, prediction in metrics['ml_predictions'].items():
            # Format criterion name for display
            display_name = criterion.replace('_', ' ').title()
            
            if prediction.get('success', False):
                # Successful prediction - format for display
                formatted_pred = predictor.format_prediction_for_display(prediction)
                formatted['ml_form_checks'][criterion] = formatted_pred
            else:
                # Failed prediction - show error
                formatted['ml_form_checks'][criterion] = {
                    'status': None,
                    'label': f"{display_name} (AI)",
                    'value': 'Error',
                    'confidence': 'N/A',
                    'probability': None,
                    'ai_powered': True,
                    'error': prediction.get('error', 'Prediction failed')
                }
        
        # Add overall ML assessment
        try:
            overall = predictor.get_overall_assessment(metrics['ml_predictions'])
            formatted['ml_overall'] = overall
        except Exception as e:
            logger.error(f"Error getting overall assessment: {e}")
            formatted['ml_overall'] = {
                'overall_correct': False,
                'criteria_passed': 0,
                'criteria_total': 0,
                'confidence': 'N/A',
                'message': 'Assessment unavailable'
            }
    
    # === Detailed Metrics (existing code) ===
    formatted['detailed'] = {
        'timing': metrics.get('timing'),
        'range_of_motion': metrics.get('range_of_motion'),
        'plank': metrics.get('plank'),
        'head': metrics.get('head')
    }
    
    return formatted


def process_all_repetitions(signals, visibility_scores, repetitions, fps) -> List[Dict[str, Any]]:
    """
    Process all repetitions with ML predictions.
    
    Returns:
        List of formatted metrics for each repetition
    """
    
    all_metrics = []
    
    for idx, rep in enumerate(repetitions):
        logger.info(f"Processing repetition {idx + 1}/{len(repetitions)}")
        
        # Get raw metrics + ML predictions
        raw_metrics = aggregate_repetition_metrics(signals, visibility_scores, rep, fps)
        
        # Format for display
        formatted_metrics = format_metrics_for_display(raw_metrics)
        
        # Add repetition index
        formatted_metrics['rep_number'] = idx + 1
        formatted_metrics['raw'] = raw_metrics  # Keep raw data
        
        all_metrics.append(formatted_metrics)
    
    return all_metrics


def calculate_overall_statistics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate overall statistics across all repetitions.
    Now includes ML prediction statistics.
    
    Returns:
        Dictionary with overall stats including ML performance
    """
    
    if not all_metrics:
        return {}
    
    total_reps = len(all_metrics)
    
    # === Rule-based form checks (existing code) ===
    form_checks = {
        'full_depth_count': 0,
        'full_lockout_count': 0,
        'complete_rep_count': 0,
        'good_hip_alignment_count': 0,
        'good_body_position_count': 0,
        'good_head_position_count': 0
    }
    
    # Average timing
    total_time = 0
    
    for metrics in all_metrics:
        # Rule-based checks
        if metrics['form_checks'].get('full_depth', {}).get('status'):
            form_checks['full_depth_count'] += 1
        if metrics['form_checks'].get('full_lockout', {}).get('status'):
            form_checks['full_lockout_count'] += 1
        if metrics['form_checks'].get('complete_rep', {}).get('status'):
            form_checks['complete_rep_count'] += 1
        if metrics['form_checks'].get('hip_alignment', {}).get('status'):
            form_checks['good_hip_alignment_count'] += 1
        if metrics['form_checks'].get('body_straight', {}).get('status'):
            form_checks['good_body_position_count'] += 1
        if metrics['form_checks'].get('head_position', {}).get('status'):
            form_checks['good_head_position_count'] += 1
        
        # Timing
        if metrics['raw'].get('timing'):
            total_time += metrics['raw']['timing']['rep_time']
    
    # === NEW: ML prediction statistics ===
    ml_stats = {
        'rom_correct_count': 0,
        'hips_correct_count': 0,
        'head_correct_count': 0,
        'all_criteria_correct_count': 0,
        'high_confidence_count': 0,
        'predictions_failed': 0
    }
    
    for metrics in all_metrics:
        if metrics.get('ml_form_checks'):
            # Count correct predictions per criterion
            rom_pred = metrics['ml_form_checks'].get('range_of_motion', {})
            if rom_pred.get('success') and rom_pred.get('status'):
                ml_stats['rom_correct_count'] += 1
            
            hips_pred = metrics['ml_form_checks'].get('hips', {})
            if hips_pred.get('success') and hips_pred.get('status'):
                ml_stats['hips_correct_count'] += 1
            
            head_pred = metrics['ml_form_checks'].get('head_position', {})
            if head_pred.get('success') and head_pred.get('status'):
                ml_stats['head_correct_count'] += 1
            
            # All criteria correct
            if metrics.get('ml_overall', {}).get('overall_correct'):
                ml_stats['all_criteria_correct_count'] += 1
            
            # High confidence predictions
            if metrics.get('ml_overall', {}).get('confidence') == 'high':
                ml_stats['high_confidence_count'] += 1
        else:
            ml_stats['predictions_failed'] += 1
    
    return {
        'total_reps': total_reps,
        'complete_reps': form_checks['complete_rep_count'],
        'completion_rate': f"{(form_checks['complete_rep_count'] / total_reps * 100):.1f}%",
        'average_rep_time': f"{(total_time / total_reps):.2f}s" if total_reps > 0 else "N/A",
        
        # Rule-based summary
        'form_checks_summary': {
            'full_depth': f"{form_checks['full_depth_count']}/{total_reps}",
            'full_lockout': f"{form_checks['full_lockout_count']}/{total_reps}",
            'hip_alignment': f"{form_checks['good_hip_alignment_count']}/{total_reps}",
            'body_position': f"{form_checks['good_body_position_count']}/{total_reps}",
            'head_position': f"{form_checks['good_head_position_count']}/{total_reps}"
        },
        
        # NEW: ML prediction summary
        'ml_summary': {
            'range_of_motion': f"{ml_stats['rom_correct_count']}/{total_reps}",
            'hips': f"{ml_stats['hips_correct_count']}/{total_reps}",
            'head_position': f"{ml_stats['head_correct_count']}/{total_reps}",
            'all_correct': f"{ml_stats['all_criteria_correct_count']}/{total_reps}",
            'high_confidence': f"{ml_stats['high_confidence_count']}/{total_reps}",
            'success_rate': f"{((total_reps - ml_stats['predictions_failed']) / total_reps * 100):.1f}%"
        }
    }