import os
import shutil
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.conf import settings
from .models import PushupVideosModel
from .processors.uploading_processor import UploadingProcessor



def home(request):
    """Home page view"""
    return render(request, "index.html")


def upload_video(request):
    """Handle video upload and processing"""
    if request.method == "POST":
        video_file = request.FILES.get('video')
        
        if not video_file:
            return JsonResponse({"error": "No video file provided"}, status=400)

        # Save to database
        video_obj = PushupVideosModel.objects.create(video=video_file)
        video_path = video_obj.video.path
        
        # Process video
        processor = UploadingProcessor()
        results = processor._process_video(video_path)
        
        # Store results in session
        request.session['analysis_results'] = {
            "video_id": video_obj.id,
            "total_reps": results.get('total_reps', 0),
            "output_dir": results.get('output_dir', '')
        }
        
        return JsonResponse({
            "status": "success",
            "redirect_url": "/demo/results/"
        })
    
    return render(request, "uploading_file/upload_video.html")


def results_view(request):
    """Display processing results"""
    results_data = request.session.get('analysis_results')
    
    if not results_data:
        return redirect('/demo/upload/')
    
    # Get data from session
    total_reps = results_data.get('total_reps', 0)
    output_dir = results_data.get('output_dir', '')
    
    # Build full path to scan directory
    output_path = os.path.join(settings.MEDIA_ROOT, output_dir)
    
    # Collect video clips
    repetition_clips = []
    if os.path.exists(output_path):
        for filename in sorted(os.listdir(output_path)):
            if filename.endswith('.mp4'):
                repetition_clips.append({
                    'filename': filename,
                    'path': f'/media/{output_dir}/{filename}'
                })
    
    # Pass data directly to template
    context = {
        'total_reps': total_reps,
        'repetition_clips': repetition_clips
    }
    
    return render(request, "uploading_file/results_view.html", context)

