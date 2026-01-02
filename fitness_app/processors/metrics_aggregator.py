from typing import Dict, List, Any
from .report_repetition_tools_basic import get_timing, get_rom
from .report_repetition_tools_plank import get_plank
from .report_repetition_tools_head import get_head


def aggregate_repetition_metrics(signals, visibility_scores, repetition, fps) -> Dict[str, Any]:
    
    try:
        timing = get_timing(signals, visibility_scores, repetition, fps)
    except Exception as e:
        print(f"Error getting timing metrics: {e}")
        timing = None
    
    try:
        rom = get_rom(signals, visibility_scores, repetition, fps)
    except Exception as e:
        print(f"Error getting ROM metrics: {e}")
        rom = None
    
    try:
        plank = get_plank(signals, visibility_scores, repetition, fps)
    except Exception as e:
        print(f"Error getting plank metrics: {e}")
        plank = None
    
    try:
        head = get_head(signals, visibility_scores, repetition)
    except Exception as e:
        print(f"Error getting head metrics: {e}")
        head = None
    
    return {
        'timing': timing,
        'range_of_motion': rom,
        'plank': plank,
        'head': head,
        'frame_info': {
            'start_frame': repetition['start_frame'],
            'end_frame': repetition['end_frame'],
            'bottom_frame': repetition['bottom_frame']
        }
    }


def format_metrics_for_display(metrics: Dict[str, Any]) -> Dict[str, Any]:

    formatted = {
        'summary': {},
        'detailed': {},
        'form_checks': {}
    }
    
    # === Summary Metrics (for cards/overview) ===
    if metrics.get('timing'):
        formatted['summary']['total_time'] = f"{metrics['timing']['rep_time']:.2f}s"
        formatted['summary']['down_time'] = f"{metrics['timing']['down_time']:.2f}s"
        formatted['summary']['up_time'] = f"{metrics['timing']['up_time']:.2f}s"
    
    if metrics.get('range_of_motion'):
        formatted['summary']['rom'] = f"{metrics['range_of_motion']['range_of_motion']:.1f}°"
        formatted['summary']['min_angle'] = f"{metrics['range_of_motion']['min_elbow_angle']:.1f}°"
        formatted['summary']['max_angle'] = f"{metrics['range_of_motion']['max_elbow_angle']:.1f}°"
    
    # === Form Checks (pass/fail indicators) ===
    if metrics.get('range_of_motion'):
        formatted['form_checks']['full_depth'] = {
            'status': metrics['range_of_motion']['full_depth'],
            'label': 'Full Depth',
            'value': 'Correct' if metrics['range_of_motion']['full_depth'] else 'Incorrect'
        }
        formatted['form_checks']['full_lockout'] = {
            'status': metrics['range_of_motion']['full_lockout'],
            'label': 'Full Lockout',
            'value': 'Correct' if metrics['range_of_motion']['full_lockout'] else 'Incorrect'
        }
        formatted['form_checks']['complete_rep'] = {
            'status': metrics['range_of_motion']['is_complete_rep'],
            'label': 'Complete Rep',
            'value': 'Yes' if metrics['range_of_motion']['is_complete_rep'] else 'No'
        }
    
    if metrics.get('plank'):
        # Hip alignment check (lower deviation = better)
        hip_deviation = metrics['plank']['hip_deviation_mean_abs']
        formatted['form_checks']['hip_alignment'] = {
            'status': hip_deviation < 10,  # Threshold: 10cm
            'label': 'Hip Alignment',
            'value': f"{hip_deviation:.1f}cm deviation"
        }
        
        # Body angle check (should be straight, ~180°)
        body_angle = metrics['plank']['body_angle_mean_deg']
        formatted['form_checks']['body_straight'] = {
            'status': 160 <= body_angle <= 200,
            'label': 'Body Position',
            'value': 'Straight' if 160 <= body_angle <= 200 else 'Not Straight'
        }
    
    if metrics.get('head'):
        # Head position check (neutral = ~160-180°)
        head_angle = metrics['head']['torso_head_angle_mean']
        formatted['form_checks']['head_position'] = {
            'status': 150 <= head_angle <= 190,
            'label': 'Head Position',
            'value': 'Neutral' if 150 <= head_angle <= 190 else 'Not Neutral'
        }
    
    # === Detailed Metrics (for expanded view) ===
    formatted['detailed'] = {
        'timing': metrics.get('timing'),
        'range_of_motion': metrics.get('range_of_motion'),
        'plank': metrics.get('plank'),
        'head': metrics.get('head')
    }
    
    return formatted


def process_all_repetitions(signals, visibility_scores, repetitions, fps) -> List[Dict[str, Any]]:

    all_metrics = []
    
    for idx, rep in enumerate(repetitions):
        # Get raw metrics
        raw_metrics = aggregate_repetition_metrics(signals, visibility_scores, rep, fps)
        
        # Format for display
        formatted_metrics = format_metrics_for_display(raw_metrics)
        
        # Add repetition index
        formatted_metrics['rep_number'] = idx + 1
        formatted_metrics['raw'] = raw_metrics  # Keep raw data for advanced use
        
        all_metrics.append(formatted_metrics)
    
    return all_metrics


def calculate_overall_statistics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    if not all_metrics:
        return {}
    
    total_reps = len(all_metrics)
    
    # Count form checks
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
        # Form checks
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
    
    return {
        'total_reps': total_reps,
        'complete_reps': form_checks['complete_rep_count'],
        'completion_rate': f"{(form_checks['complete_rep_count'] / total_reps * 100):.1f}%",
        'average_rep_time': f"{(total_time / total_reps):.2f}s" if total_reps > 0 else "N/A",
        'form_checks_summary': {
            'full_depth': f"{form_checks['full_depth_count']}/{total_reps}",
            'full_lockout': f"{form_checks['full_lockout_count']}/{total_reps}",
            'hip_alignment': f"{form_checks['good_hip_alignment_count']}/{total_reps}",
            'body_position': f"{form_checks['good_body_position_count']}/{total_reps}",
            'head_position': f"{form_checks['good_head_position_count']}/{total_reps}"
        }
    }