from django.db import models

class PushupVideosModel(models.Model):
    video = models.FileField(upload_to='pushup_videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        rtn = self.video.name
