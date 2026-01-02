from .landmark_extraction import extract_landmarks_from_video
from .draw_landmarks_on_video import draw_landmarks_on_video
from .compute_signals import compute_pushup_signals
from .detecting_repetitions import detect_pushup_repetitions
from .video_cut import cut_video_segments
from .metrics_aggregator import process_all_repetitions, calculate_overall_statistics
import os
from django.conf import settings


class UploadingProcessor:
    def __init__(self):
        pass
    
    def _process_video(self, video_path):
        output_dir = "output_cuts"
        full_output_path = os.path.join(settings.MEDIA_ROOT, output_dir)
        os.makedirs(full_output_path, exist_ok=True)
        # full_output_path = os.path.join(output_dir)
        
        all_data, rotation, fps = extract_landmarks_from_video(video_path)
        signals, visibility_scores = compute_pushup_signals(all_data, fps)
        repetitions = detect_pushup_repetitions(signals, all_data, visibility_scores, fps)
        cut_video_segments(video_path, repetitions, full_output_path)
        
        # Process all metrics
        rep_metrics = process_all_repetitions(signals, visibility_scores, repetitions, fps)
        overall_stats = calculate_overall_statistics(rep_metrics)
        
        return {
            "total_reps": len(repetitions),
            "output_dir": output_dir,
            "repetitions": rep_metrics,
            "overall_statistics": overall_stats
        }

if __name__ == "__main__":
    video_path = "C:\\Users\\micha\\Downloads\\rncNGvYT.mp4"
    uploading_processor = UploadingProcessor()
    print(uploading_processor._process_video(video_path))