import os
import shutil
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.conf import settings
from .models import PushupVideosModel
from .processors.uploading_processor import UploadingProcessor



def home(request):
    """Home page view"""
    return render(request, "index.html")


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):  # ← Added this
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
    """Display processing results"""
    results_data = request.session.get('analysis_results')
    
    if not results_data:
        return redirect('/demo/upload/')
    
    # Get data from session
    total_reps = results_data.get('total_reps', 0)
    output_dir = results_data.get('output_dir', '')
    repetitions_data = results_data.get('repetitions', [])
    overall_statistics = results_data.get('overall_statistics', {})
    
    # Build full path to scan directory
    output_path = os.path.join(settings.MEDIA_ROOT, output_dir)
    
    # Collect video clips with their metrics
    repetition_clips = []
    if os.path.exists(output_path):
        video_files = sorted([f for f in os.listdir(output_path) if f.endswith('.mp4')])
        
        for filename in video_files:
            # Extract rep number from filename (e.g., "rep_1.mp4" -> 1)
            try:
                if 'rep_' in filename:
                    rep_part = filename.split('rep_')[1]  # Gets "1.mp4" from "filename_rep_1.mp4"
                    rep_number = int(rep_part.split('.')[0])  # Gets 1
            except (IndexError, ValueError):
                continue
            
            # Find matching metrics for this rep
            rep_metrics = next(
                (rep for rep in repetitions_data if rep.get('rep_number') == rep_number),
                None
            )
            
            if rep_metrics:
                repetition_clips.append({
                    'filename': filename,
                    'video_url': f'/media/{output_dir}/{filename}',
                    'rep_number': rep_number,
                    'metrics': rep_metrics
                })
    
    # Pass complete data to template
    context = {
        'total_reps': total_reps,
        'repetition_clips': repetition_clips,
        'overall_statistics': overall_statistics
    }
    
    return render(request, "uploading_file/results_view.html", context)

