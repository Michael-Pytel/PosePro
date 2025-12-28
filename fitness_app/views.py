import subprocess
import os
import sys
import json
import shutil
import pandas as pd
import threading
from django.utils.safestring import mark_safe
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.conf import settings
from .models import PushupVideosModel

MAX_FILE_SIZE = 100 * 1024 * 1024

# Global dict to store progress for each video upload session
PIPELINE_PROGRESS = {}


from django.http import JsonResponse

# Add this function to your views.py
def get_progress(request):
    """
    API endpoint to return current pipeline processing progress
    """
    # You should have a global or session variable tracking progress
    # For now, return a basic structure
    
    # If you have a session-based progress tracker:
    progress_data = request.session.get('PIPELINE_PROGRESS', {
        'current_stage': 'idle',
        'progress': 0
    })
    
    return JsonResponse(progress_data)
    
def home(request):
    """Home page view"""
    return render(request, "index.html")

def update_pipeline_progress(session_id, stage, progress):
    """Update progress for a specific pipeline stage"""
    if session_id not in PIPELINE_PROGRESS:
        PIPELINE_PROGRESS[session_id] = {}
    
    PIPELINE_PROGRESS[session_id] = {
        'current_stage': stage,
        'progress': progress,
        'timestamp': pd.Timestamp.now().isoformat()
    }

def get_pipeline_progress(request):
    """API endpoint to get current pipeline progress"""
    session_id = request.session.session_key
    
    if session_id in PIPELINE_PROGRESS:
        return JsonResponse(PIPELINE_PROGRESS[session_id])
    
    return JsonResponse({'current_stage': 'idle', 'progress': 0})

def process_video_with_tracking(session_id, script_path, folder_path, output_path):
    """Process video and track pipeline progress"""
    
    # Pipeline stages in order
    stages = [
        ('segmentation', 'Segmenting repetitions...'),
        ('landmarks', 'Extracting pose landmarks...'),
        ('angles', 'Calculating joint angles...'),
        ('aggregation', 'Aggregating features...'),
        ('prediction', 'Making predictions...')
    ]
    
    try:
        # Start with segmentation
        update_pipeline_progress(session_id, stages[0][0], 0)
        
        process = subprocess.Popen(
            [
                sys.executable,
                script_path,
                "-i", folder_path,
                "-o", output_path,
                "-e", "pushup",
                "--debug",
                "-p", "test",
                "--visualize",
                "--no-confirm"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        
        stage_index = 0
        for line in process.stdout:
            print("PROCESS:", line.strip())
            
            # Track progress based on output messages
            line_lower = line.lower()
            
            if "ekstrakcja landmarks" in line_lower:
                stage_index = 1
                update_pipeline_progress(session_id, stages[1][0], 20)
            elif "kąty" in line_lower or "calculating angles" in line_lower:
                stage_index = 2
                update_pipeline_progress(session_id, stages[2][0], 40)
            elif "agregacja" in line_lower or "aggregat" in line_lower:
                stage_index = 3
                update_pipeline_progress(session_id, stages[3][0], 60)
            elif "predykcja" in line_lower or "prediction" in line_lower:
                stage_index = 4
                update_pipeline_progress(session_id, stages[4][0], 80)
            
            # Update progress based on stage
            if stage_index < len(stages):
                if stage_index == 0:
                    update_pipeline_progress(session_id, stages[0][0], 10)
        
        stdout, stderr = process.communicate()
        print("=== STDOUT ===")
        print(stdout)
        print("=== STDERR ===")
        print(stderr)
        
        # Mark as complete
        update_pipeline_progress(session_id, 'complete', 100)
        
    except Exception as e:
        print(f"Error in process_video_with_tracking: {e}")
        update_pipeline_progress(session_id, 'error', 0)


def upload_video(request):
    if request.method == "POST":
        session_id = request.session.session_key
        files = request.FILES.getlist("video")
        #Deleting previous videos
        athlete_videos_dir = os.path.join(settings.MEDIA_ROOT, "athlete_videos")

        if os.path.exists(athlete_videos_dir):
            shutil.rmtree(athlete_videos_dir)

        os.makedirs(athlete_videos_dir, exist_ok=True)
        for f in files:
            obj = PushupVideosModel.objects.create(video=f)
            folder_path = os.path.join(settings.MEDIA_ROOT, "videos", str(obj.id))
            os.makedirs(folder_path, exist_ok=True)

            new_video_path = os.path.join(folder_path, f.name)

            with open(new_video_path, "wb+") as dest:
                for chunk in f.chunks():
                    dest.write(chunk)

            script_path = os.path.join(settings.BASE_DIR, "static/backend-scripts/exercise_processor/working_pipeline/run_exercise_processor.py")
            print("### BEFORE POPEN ###")

            # Run video processing in a thread to allow progress updates
            processing_thread = threading.Thread(
                target=process_video_with_tracking,
                args=(session_id, script_path, folder_path, "./media/athlete_videos")
            )
            processing_thread.daemon = True
            processing_thread.start()
            
            # Wait for processing to complete
            processing_thread.join()
            
            videos_dir = os.path.join("./media/athlete_videos", "videos")
            viz_path = None
            rep_clips = []

            if os.path.exists(videos_dir):
                for filename in os.listdir(videos_dir):
                    file_path = os.path.join(videos_dir, filename)
                    relative_path = os.path.relpath(file_path, settings.MEDIA_ROOT)

                    if "_FULL_visualization" in filename:
                        viz_path = f"/media/{relative_path}"
                    elif filename.endswith(".mp4"):
                        rep_clips.append({
                            "filename": filename,
                            "path": f"/media/{relative_path}"
                        })
            
            # Load predictions
            predictions_data = []
            predictions_csv = os.path.join("./media", "reps_predictions.csv")
            
            if os.path.exists(predictions_csv):
                try:
                    df_predictions = pd.read_csv(predictions_csv)
                    
                    # Convert DataFrame to list of dicts for template
                    for idx, row in df_predictions.iterrows():
                        pred_dict = {
                            "id": row.get('id', f'rep_{idx}'),
                            "head_position": row.get('pred_head_position_model', 'N/A'),
                            "hips": row.get('pred_hips_model', 'N/A'),
                            "elbows": row.get('pred_elbows_model', 'N/A'),
                            "range_of_motion": row.get('pred_range_of_motion_model', 'N/A'),
                            "all_models": row.get('pred_all_models', 'N/A')
                        }
                        predictions_data.append(pred_dict)
                except Exception as e:
                    print(f"Error loading predictions: {e}")
            
            request.session['analysis_results'] = {
                "video_id": obj.id,
                "visualization_video": viz_path,
                "repetition_clips": rep_clips,
                "total_reps": len(rep_clips),
                "predictions": predictions_data
            }

            return JsonResponse({
                "status": "success",
                "message": "Video processed successfully",
                "redirect_url": "/results/"
            })
        
        return redirect("/results/")
    return render(request, "uploading_file/upload_video.html")


def results_view(request):
    results_data = request.session.get('analysis_results')

    if not results_data:
        return redirect('/upload/')

    results_json = mark_safe(json.dumps(results_data))

    return render(request, "uploading_file/results_view.html", {'results_data': results_json})