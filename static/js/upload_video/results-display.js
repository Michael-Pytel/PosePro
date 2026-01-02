// Simple video interaction effects
document.addEventListener('DOMContentLoaded', function() {
    console.log('Results page loaded');
    
    // Animate stat cards on load
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach((card, index) => {
        card.style.animationDelay = `${0.4 + (index * 0.1)}s`;
    });
    
    // Add play/pause interaction feedback for videos
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
        video.addEventListener('play', function() {
            this.closest('.clip-card')?.classList.add('playing');
        });
        
        video.addEventListener('pause', function() {
            this.closest('.clip-card')?.classList.remove('playing');
        });
    });
});