import os
import shutil
from django.conf import settings
from django.utils.deprecation import MiddlewareMixin
from .models import PushupVideosModel


class DemoCleanupMiddleware(MiddlewareMixin):
    """
    Middleware to clean up demo files before processing new uploads
    """
    
    def process_request(self, request):
        """
        Called on every request before the view is executed
        """
        # Only cleanup when someone is about to upload (POST to demo/upload)
        if request.path == '/demo/upload/' and request.method == 'POST':
            self.cleanup_demo_files()
        
        return None  
    
    def cleanup_demo_files(self):
        """Clean up temporary demo files"""
        
        # 1. Delete all uploaded videos from database and filesystem
        try:
            videos = PushupVideosModel.objects.all()
            for video in videos:
                if video.video and os.path.exists(video.video.path):
                    os.remove(video.video.path)
                    video_dir = os.path.dirname(video.video.path)
                    if os.path.exists(video_dir) and not os.listdir(video_dir):
                        os.rmdir(video_dir)
            videos.delete()
        except Exception as e:
            print(f"Error cleaning up videos: {e}")
        
        # 2. Clear output_cuts directory
        output_cuts_path = os.path.join(settings.MEDIA_ROOT, "output_cuts")
        if os.path.exists(output_cuts_path):
            try:
                shutil.rmtree(output_cuts_path)
                os.makedirs(output_cuts_path, exist_ok=True)
            except Exception as e:
                print(f"Error cleaning output_cuts: {e}")
        
        print("Demo files cleaned up via middleware")