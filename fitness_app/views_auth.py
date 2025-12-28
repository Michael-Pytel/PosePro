from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Sum, Count
from .forms import RegisterForm, LoginForm
from .models import User, PushupVideosModel, ExerciseAnalysis

def register_view(request):
    """User registration view"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome to PosePro Coach, {user.username}!')
            return redirect('dashboard')
    else:
        form = RegisterForm()
    
    return render(request, 'auth/register.html', {'form': form})

def login_view(request):
    """User login view"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {user.username}!')
                return redirect('dashboard')
    else:
        form = LoginForm()
    
    return render(request, 'auth/login.html', {'form': form})

def logout_view(request):
    """User logout view"""
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('home')

@login_required
def dashboard_view(request):
    """User dashboard showing exercise statistics - ONLY for logged-in users"""
    user = request.user
    
    # Get all videos for this user (exclude anonymous uploads)
    user_videos = PushupVideosModel.objects.filter(user=user)
    
    # Calculate statistics
    total_videos = user_videos.count()
    total_reps = user_videos.aggregate(Sum('total_reps'))['total_reps__sum'] or 0
    correct_reps = user_videos.aggregate(Sum('correct_reps'))['correct_reps__sum'] or 0
    
    # Calculate average accuracy
    if total_reps > 0:
        average_accuracy = round((correct_reps / total_reps) * 100, 1)
    else:
        average_accuracy = 0
    
    # Get recent videos (last 6)
    recent_videos = user_videos.order_by('-uploaded_at')[:6]
    
    context = {
        'total_videos': total_videos,
        'total_reps': total_reps,
        'correct_reps': correct_reps,
        'average_accuracy': average_accuracy,
        'recent_videos': recent_videos,
    }
    
    return render(request, 'dashboard.html', context)

@login_required
def video_detail_view(request, video_id):
    """View detailed analysis for a specific video"""
    video = get_object_or_404(PushupVideosModel, id=video_id, user=request.user)
    
    # Get analysis if it exists
    try:
        analysis = video.analysis
    except ExerciseAnalysis.DoesNotExist:
        analysis = None
    
    context = {
        'video': video,
        'analysis': analysis,
    }
    
    return render(request, 'video_detail.html', context)