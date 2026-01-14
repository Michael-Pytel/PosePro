import os
import shutil
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.conf import settings
from .models import PushupVideosModel
from fitness_app.uploading_processor import UploadingProcessor



def home(request):
    """Home page view"""
    return render(request, "index.html")


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if obj is None:  # NEW - handle None explicitly
        return None
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        # Handle NaN
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8, bool)):  # Include native bool
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def upload_video(request):
    """Handle video upload and processing"""
    if request.method == "POST":
        video_file = request.FILES.get('video')
        
        if not video_file:
            return JsonResponse({"error": "No video file provided"}, status=400)

        try:
            # Save to database
            video_obj = PushupVideosModel.objects.create(video=video_file)
            video_path = video_obj.video.path
            
            # Process video
            processor = UploadingProcessor()
            results = processor._process_video(video_path)
            
            if results.get('repetitions'):
                first_rep = results['repetitions'][0]
                ml_checks = first_rep.get('ml_form_checks', {})
                print(f"✓ ML predictions in results: {list(ml_checks.keys())}")
                if ml_checks.get('range_of_motion'):
                    print(f"  ROM prediction: {ml_checks['range_of_motion'].get('value')}")

            # Convert numpy types to native Python types
            results_clean = convert_numpy_types(results)
            
            # Store ALL results in session (including metrics!)
            request.session['analysis_results'] = {
                "video_id": video_obj.id,
                "total_reps": results_clean.get('total_reps', 0),
                "output_dir": results_clean.get('output_dir', ''),
                "repetitions": results_clean.get('repetitions', []),
                "overall_statistics": results_clean.get('overall_statistics', {})
            }
            
            return JsonResponse({
                "status": "success",
                "redirect_url": "/demo/results/"
            })
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error processing video: {error_trace}")
            return JsonResponse({
                "error": f"Processing failed: {str(e)}"
            }, status=500)
    
    return render(request, "uploading_file/upload_video.html")


def results_view(request):
    """Display AI processing results"""
    results_data = request.session.get('analysis_results')
    
    if not results_data:
        return redirect('/demo/upload/')
    
    # 1. Basic Data
    output_dir = results_data.get('output_dir', '')
    repetitions = results_data.get('repetitions', [])
    total_reps = len(repetitions)
    
    # 2. Compute AI Statistics
    stats = {
        'total_reps': total_reps,
        'perfect_reps': 0,
        'correct_counts': {'rom': 0, 'hips': 0, 'head': 0},
        'high_confidence_count': 0
    }

    for rep in repetitions:
        preds = rep.get('predictions', {})
        
        # Check if this rep is "Perfect" (All available models say Correct)
        is_perfect = True
        has_predictions = False
        
        for key, data in preds.items():
            if data:
                has_predictions = True
                if data.get('is_correct'):
                    stats['correct_counts'][key] = stats['correct_counts'].get(key, 0) + 1
                else:
                    is_perfect = False
                
                # Count high confidence (> 0.85)
                if data.get('confidence', 0) > 0.85:
                    stats['high_confidence_count'] += 1
        
        if has_predictions and is_perfect:
            stats['perfect_reps'] += 1

    # Calculate percentages for the UI
    stats['success_rate'] = int((stats['perfect_reps'] / total_reps * 100) if total_reps > 0 else 0)

    # 3. Match Videos to Reps
    output_path = os.path.join(settings.MEDIA_ROOT, output_dir)
    repetition_clips = []
    
    if os.path.exists(output_path):
        video_files = [f for f in os.listdir(output_path) if f.endswith('.mp4')]
    
        # Sort numerically by rep number
        def get_rep_number(filename):
            try:
                if 'rep_' in filename:
                    rep_part = filename.split('rep_')[1]
                    return int(rep_part.split('.')[0])
                return 0
            except (IndexError, ValueError):
                return 0
        
        video_files = sorted(video_files, key=get_rep_number)
        
        for filename in video_files:
            try:
                # Extract rep ID from "rep_1.mp4"
                if 'rep_' in filename:
                    rep_part = filename.split('rep_')[1]
                    rep_id = int(rep_part.split('.')[0])
                    
                    # Find the matching rep object from session
                    metrics = next((r for r in repetitions if r.get('rep_id') == rep_id), None)
                    
                    if metrics:
                        # Add timing data to metrics if available
                        # Assuming your processor adds timing to features['timing']
                        timing_data = metrics.get('features', {}).get('timing', {})
                        
                        # Merge timing into the main metrics dict for easy template access
                        metrics_with_timing = {**metrics}
                        if timing_data:
                            metrics_with_timing['up_time'] = timing_data.get('up_time')
                            metrics_with_timing['down_time'] = timing_data.get('down_time')
                            metrics_with_timing['bottom_pause'] = timing_data.get('bottom_pause')
                        
                        repetition_clips.append({
                            'rep_number': rep_id,
                            'filename': filename,
                            'video_url': f'{settings.MEDIA_URL}{output_dir}/{filename}',
                            'metrics': metrics_with_timing  # Use the enhanced metrics
                        })
            except (IndexError, ValueError):
                continue

    context = {
        'overall_statistics': stats,
        'repetition_clips': repetition_clips,
    }
    
    return render(request, "uploading_file/results_view.html", context)
