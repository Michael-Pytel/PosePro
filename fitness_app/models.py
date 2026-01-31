from django.db import models

class PushupVideosModel(models.Model):
    """Video upload model"""
    video = models.FileField(upload_to='pushup_videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    
    # Store analysis results
    total_reps = models.IntegerField(default=0)
    correct_reps = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Video {self.id} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def accuracy_percentage(self):
        if self.total_reps == 0:
            return 0
        return round((self.correct_reps / self.total_reps) * 100, 1)