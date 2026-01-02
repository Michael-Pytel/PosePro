from .landmark_extraction import extract_landmarks_from_video
from .draw_landmarks_on_video import draw_landmarks_on_video
from .compute_signals import compute_pushup_signals
from .detecting_repetitions import detect_pushup_repetitions
from .video_cut import cut_video_segments
from .report_repetition_tools_advanced import test
import os
from django.conf import settings

class UploadingProcessor:
    def __init__(self):
        pass
    def _process_video(self, video_path):
        output_dir = os.path.join(settings.MEDIA_ROOT, "output_cuts")
        all_data, rotation, fps = extract_landmarks_from_video(video_path)
        # Function below is only for testing not for final solution
        # draw_landmarks_on_video(video_path, all_data, output_dir)
        signals, visibility_scores = compute_pushup_signals(all_data, fps)
        repetitions = detect_pushup_repetitions(signals, all_data, visibility_scores, fps)
        cut_video_segments(video_path, repetitions, output_dir)
        
        
        # rep_data = []
        # for rep in repetitions:
        #     result = test(signals, visibility_scores, rep, fps)
        #     rep_data.append(result)  # ← What does test() return?
        
        # Return structured data
        return {
            "total_reps": len(repetitions),
            "output_dir": output_dir
        }

if __name__ == "__main__":
    video_path = "C:\\Users\\micha\\Downloads\\rncNGvYT.mp4"
    uploading_processor = UploadingProcessor()
    uploading_processor._process_video(video_path)