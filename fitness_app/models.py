from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    """Custom User model"""
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.username

class PushupVideosModel(models.Model):
    """Video upload model linked to user"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='videos', null=True, blank=True)
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

class ExerciseAnalysis(models.Model):
    """Store detailed analysis for each video"""
    video = models.OneToOneField(PushupVideosModel, on_delete=models.CASCADE, related_name='analysis')
    
    # Paths to generated files
    visualization_video_path = models.CharField(max_length=500, blank=True)
    predictions_csv_path = models.CharField(max_length=500, blank=True)
    
    # Summary statistics
    total_repetitions = models.IntegerField(default=0)
    correct_form_count = models.IntegerField(default=0)
    
    # Detailed predictions (stored as JSON)
    predictions_data = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Analysis for Video {self.id}"