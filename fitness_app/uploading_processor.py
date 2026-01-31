from fitness_app.core.landmark_extraction import extract_landmarks_from_video
from fitness_app.core.compute_signals import compute_pushup_signals
from fitness_app.core.detecting_repetitions import detect_pushup_repetitions
from fitness_app.utils.video_cut import cut_video_segments
from fitness_app.core.feature_extractor import FeatureExtractor
from fitness_app.core.predictor import Predictor
from fitness_app.utils.progress_tracker import ProgressTracker, ProcessingStage

import os
from django.conf import settings

class UploadingProcessor:
    def __init__(self, session_id=None):
        self.feature_extractor = FeatureExtractor()
        self.predictor = Predictor()
        self.progress_tracker = ProgressTracker(session_id)

    def _process_video(self, video_path):
        output_dir = "output_cuts"
        full_output_path = os.path.join(settings.MEDIA_ROOT, output_dir)
        os.makedirs(full_output_path, exist_ok=True)

        # 1. Extract Landmarks
        self.progress_tracker.update(ProcessingStage.EXTRACTING_LANDMARKS)
        all_data, rotation, fps = extract_landmarks_from_video(video_path)
        
        # 2. Compute Signals
        self.progress_tracker.update(ProcessingStage.COMPUTING_SIGNALS)
        signals, visibility_scores = compute_pushup_signals(all_data, fps)
        
        # 3. Detect Repetitions
        self.progress_tracker.update(ProcessingStage.DETECTING_REPS)
        repetitions = detect_pushup_repetitions(signals, all_data, visibility_scores, fps)

        # 4. Feature Extraction & Prediction Loop
        total_reps = len(repetitions)
        for idx, rep in enumerate(repetitions, 1):
            
            if idx <= total_reps // 2:
                self.progress_tracker.update_with_substep(
                    ProcessingStage.EXTRACTING_FEATURES, idx, total_reps
                )
            else:
                self.progress_tracker.update_with_substep(
                    ProcessingStage.MAKING_PREDICTIONS, idx, total_reps
                )
            # A. Extract Features
            ui_features, model_inputs = self.feature_extractor.extract_features(
                signals, visibility_scores, rep, fps
            )
            
            # B. Assign to rep
            rep['features'] = ui_features
            rep['predictions'] = self.predictor.predict_repetition(model_inputs)

        # 5. Cut Video Segments
        self.progress_tracker.update(ProcessingStage.CUTTING_VIDEOS)
        cut_video_segments(video_path, repetitions, full_output_path)
        
        # 6. Complete
        self.progress_tracker.update(ProcessingStage.COMPLETE)

        return {
            "total_reps": len(repetitions),
            "output_dir": output_dir,
            "repetitions": repetitions,
            "session_id": self.progress_tracker.session_id
        }
