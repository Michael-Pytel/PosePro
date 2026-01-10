from fitness_app.core.landmark_extraction import extract_landmarks_from_video
from fitness_app.core.compute_signals import compute_pushup_signals
from fitness_app.core.detecting_repetitions import detect_pushup_repetitions
from fitness_app.utils.video_cut import cut_video_segments
from fitness_app.core.feature_extractor import FeatureExtractor
from fitness_app.core.predictor import Predictor

import os
from django.conf import settings

class UploadingProcessor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.predictor = Predictor()

    def _process_video(self, video_path):
        output_dir = "output_cuts"
        full_output_path = os.path.join(settings.MEDIA_ROOT, output_dir)
        os.makedirs(full_output_path, exist_ok=True)

        # 1. Pipeline
        all_data, rotation, fps = extract_landmarks_from_video(video_path)
        signals, visibility_scores = compute_pushup_signals(all_data, fps)
        repetitions = detect_pushup_repetitions(signals, all_data, visibility_scores, fps)

        # 2. Prediction & Feature Loop
        for rep in repetitions:
            # A. Get EVERYTHING from the extractor in one line
            ui_features, model_inputs = self.feature_extractor.extract_features(
                signals, visibility_scores, rep, fps
            )
            
            # B. Assign to rep
            rep['features'] = ui_features
            rep['predictions'] = self.predictor.predict_repetition(model_inputs)

        # 3. Finalize
        cut_video_segments(video_path, repetitions, full_output_path)

        return {
            "total_reps": len(repetitions),
            "output_dir": output_dir,
            "repetitions": repetitions, 
        }

if __name__ == "__main__":
    # --- 1. Setup Django Environment ---
    import os
    import django
    from django.conf import settings

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fitness_app.settings')
    django.setup()


    # --- 3. Run the Processor ---
    video_path = "C:\\Users\\micha\\Downloads\\recordings\\recordings\\own_recordings\\pushups\\pushup_02.mp4"
    uploading_processor = UploadingProcessor()
    
    # Use pprint for readable output
    import pprint
    pprint.pprint(uploading_processor._process_video(video_path))