"""
Progress tracking utility for video processing pipeline.
Stores progress in Django cache for real-time updates.
"""
from django.core.cache import cache
from enum import Enum
from typing import Optional
import uuid


class ProcessingStage(Enum):
    """Enumeration of processing stages"""
    UPLOADING = ("Uploading video...", 0)
    EXTRACTING_LANDMARKS = ("Extracting body landmarks...", 15)
    COMPUTING_SIGNALS = ("Computing movement signals...", 35)
    DETECTING_REPS = ("Detecting repetitions...", 55)
    EXTRACTING_FEATURES = ("Extracting biomechanical features...", 70)
    MAKING_PREDICTIONS = ("Analyzing exercise form...", 85)
    CUTTING_VIDEOS = ("Preparing video segments...", 95)
    COMPLETE = ("Processing complete!", 100)
    
    def __init__(self, message, progress):
        self.message = message
        self.progress = progress


class ProgressTracker:
    """Tracks progress of video processing operations"""
    
    def __init__(self, session_id: Optional[str] = None):
    
        self.session_id = session_id or str(uuid.uuid4())
        self.cache_key = f"progress_{self.session_id}"
        self.cache_timeout = 3600  # 1 hour
        
    def update(self, stage: ProcessingStage, details: Optional[str] = None):
        """
        Update progress for current processing stage
        
        Args:
            stage: Current processing stage
            details: Optional additional details (e.g., "Rep 3/10")
        """
        progress_data = {
            'stage': stage.name,
            'message': stage.message,
            'progress': stage.progress,
            'details': details,
            'session_id': self.session_id
        }
        cache.set(self.cache_key, progress_data, self.cache_timeout)
        
    def update_with_substep(self, stage: ProcessingStage, current: int, total: int):
        """
        Update progress with substep information
        
        Args:
            stage: Current processing stage
            current: Current item number
            total: Total number of items
        """
        details = f"{current}/{total}"
        
        # Calculate interpolated progress within stage range
        if stage == ProcessingStage.COMPLETE:
            progress = 100
        else:
            # Get the next stage to determine the range
            stages = list(ProcessingStage)
            current_idx = stages.index(stage)
            next_stage = stages[current_idx + 1] if current_idx + 1 < len(stages) else stage
            
            # Interpolate between current and next stage
            stage_range = next_stage.progress - stage.progress
            substep_progress = (current / total) * stage_range if total > 0 else 0
            progress = stage.progress + substep_progress
        
        progress_data = {
            'stage': stage.name,
            'message': stage.message,
            'progress': round(progress, 1),
            'details': details,
            'session_id': self.session_id
        }
        cache.set(self.cache_key, progress_data, self.cache_timeout)
    
    def get_progress(self) -> dict:
        """
        Get current progress data
        
        Returns:
            Dictionary with progress information or None if not found
        """
        return cache.get(self.cache_key)
    
    def clear(self):
        """Clear progress data from cache"""
        cache.delete(self.cache_key)
    
    @staticmethod
    def get_by_session_id(session_id: str) -> dict:
        """
        Get progress data by session ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with progress information or None if not found
        """
        cache_key = f"progress_{session_id}"
        return cache.get(cache_key)