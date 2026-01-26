from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [
    # Home and main pages
    path('', views.home, name='home'),
    
    # Demo section
    path('demo/upload/', views.upload_video, name='demo_upload'),
    path('demo/results/', views.results_view, name='demo_results'),
    
    # API
    path('api/progress/', views.get_upload_progress, name='upload_progress'),
    path('api/check-status/', views.check_processing_status, name='check_status'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)