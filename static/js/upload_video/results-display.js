/**
 * Results Page JavaScript
 * Handles rep details toggling, metric categories, video modal, and expand/collapse all
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Results page JavaScript loaded');
    
    // ===========================
    // TOGGLE REP DETAILS
    // ===========================
    const detailToggles = document.querySelectorAll('.toggle-details');
    console.log('Found toggle buttons:', detailToggles.length);
    
    detailToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const detailsSection = document.getElementById(targetId);
            const toggleText = this.querySelector('.toggle-text');
            const toggleIcon = this.querySelector('.toggle-icon');
            
            if (detailsSection) {
                if (detailsSection.style.display === 'none' || detailsSection.style.display === '') {
                    // Expand
                    detailsSection.style.display = 'block';
                    toggleText.textContent = 'Hide Details';
                    toggleIcon.style.transform = 'rotate(180deg)';
                    this.classList.add('active');
                } else {
                    // Collapse
                    detailsSection.style.display = 'none';
                    toggleText.textContent = 'Show Details';
                    toggleIcon.style.transform = 'rotate(0deg)';
                    this.classList.remove('active');
                }
            } else {
                console.error('Details section not found for:', targetId);
            }
        });
    });
    
    // ===========================
    // EXPAND/COLLAPSE ALL BUTTON
    // ===========================
    const expandAllBtn = document.getElementById('expandAllBtn');
    let allExpanded = false;
    
    if (expandAllBtn) {
        expandAllBtn.addEventListener('click', function() {
            const allDetails = document.querySelectorAll('.rep-details');
            const allToggles = document.querySelectorAll('.toggle-details');
            
            console.log('Expand all clicked. Current state:', allExpanded);
            
            allDetails.forEach((details, index) => {
                const toggle = allToggles[index];
                if (!toggle) return;
                
                const toggleText = toggle.querySelector('.toggle-text');
                const toggleIcon = toggle.querySelector('.toggle-icon');
                
                if (!allExpanded) {
                    // Expand all
                    details.style.display = 'block';
                    toggleText.textContent = 'Hide Details';
                    toggleIcon.style.transform = 'rotate(180deg)';
                    toggle.classList.add('active');
                } else {
                    // Collapse all
                    details.style.display = 'none';
                    toggleText.textContent = 'Show Details';
                    toggleIcon.style.transform = 'rotate(0deg)';
                    toggle.classList.remove('active');
                }
            });
            
            // Toggle state and button text
            allExpanded = !allExpanded;
            this.textContent = allExpanded ? 'Collapse All Details' : 'Expand All Details';
            
            console.log('New state:', allExpanded);
        });
    }
    
    // ===========================
    // VIDEO MODAL
    // ===========================
    const videoThumbnails = document.querySelectorAll('.rep-video-thumbnail');
    const videoModal = document.getElementById('videoModal');
    const modalVideo = document.getElementById('modalVideo');
    const modalClose = document.querySelector('.video-modal-close');
    const modalOverlay = document.querySelector('.video-modal-overlay');
    
    console.log('Found video thumbnails:', videoThumbnails.length);
    console.log('Modal elements:', {
        modal: !!videoModal,
        video: !!modalVideo,
        close: !!modalClose,
        overlay: !!modalOverlay
    });
    
    // Open modal when clicking video thumbnail
    videoThumbnails.forEach(thumbnail => {
        thumbnail.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            const video = this.querySelector('video source');
            if (!video) {
                console.error('Video source not found in thumbnail');
                return;
            }
            
            // Get video URL without the #t=0.5 timestamp
            const videoSrc = video.getAttribute('src').split('#')[0];
            console.log('Opening video modal for:', videoSrc);
            
            if (!modalVideo) {
                console.error('Modal video element not found');
                return;
            }
            
            // Set video source and load
            const existingSource = modalVideo.querySelector('source');
            if (existingSource) {
                existingSource.setAttribute('src', videoSrc);
            } else {
                // Create source element if it doesn't exist
                const source = document.createElement('source');
                source.setAttribute('src', videoSrc);
                source.setAttribute('type', 'video/mp4');
                modalVideo.appendChild(source);
            }
            
            // Load the video
            modalVideo.load();
            
            // Show modal
            if (videoModal) {
                videoModal.classList.add('active');
                document.body.style.overflow = 'hidden'; // Prevent background scrolling
                
                // Auto-play after load (optional)
                modalVideo.play().catch(err => {
                    console.warn('Autoplay prevented by browser:', err);
                });
            }
        });
    });
    
    // Close modal function
    function closeModal() {
        console.log('Closing video modal');
        if (!videoModal) return;
        
        videoModal.classList.remove('active');
        if (modalVideo) {
            modalVideo.pause();
            modalVideo.currentTime = 0; // Reset video to start
        }
        document.body.style.overflow = ''; // Restore scrolling
    }
    
    // Close button
    if (modalClose) {
        modalClose.addEventListener('click', closeModal);
    }
    
    // Click overlay to close
    if (modalOverlay) {
        modalOverlay.addEventListener('click', closeModal);
    }
    
    // Escape key to close
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && videoModal && videoModal.classList.contains('active')) {
            closeModal();
        }
    });
    
    // ===========================
    // SMOOTH SCROLL TO REP (Optional feature)
    // ===========================
    // If you add jump navigation later, this will handle smooth scrolling
    function scrollToRep(repNumber) {
        const repItem = document.querySelector(`[data-rep="${repNumber}"]`);
        if (repItem) {
            repItem.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
            
            // Add highlight effect
            repItem.style.transition = 'all 0.3s ease';
            repItem.style.transform = 'scale(1.02)';
            setTimeout(() => {
                repItem.style.transform = 'scale(1)';
            }, 600);
        }
    }
    
    // ===========================
    // KEYBOARD SHORTCUTS (Optional)
    // ===========================
    document.addEventListener('keydown', function(e) {
        // Don't trigger if typing in input field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        // 'E' key to expand/collapse all
        if (e.key === 'e' || e.key === 'E') {
            if (expandAllBtn) {
                expandAllBtn.click();
            }
        }
    });
    
    // ===========================
    // LOADING ANIMATION COMPLETE
    // ===========================
    console.log('All event listeners initialized successfully');
    console.log('- Rep detail toggles:', detailToggles.length);
    console.log('- Video thumbnails:', videoThumbnails.length);
    console.log('- Expand all button:', expandAllBtn ? 'Found' : 'Not found');
});

/**
 * Utility function to format time
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string
 */
function formatTime(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(2)}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
}

/**
 * Utility function to format angle
 * @param {number} angle - Angle in degrees
 * @returns {string} Formatted angle string
 */
function formatAngle(angle) {
    return `${angle.toFixed(1)}°`;
}

/**
 * Utility function to add loading state to element
 * @param {HTMLElement} element - Element to add loading state to
 */
function addLoadingState(element) {
    element.classList.add('loading');
    element.style.opacity = '0.6';
    element.style.pointerEvents = 'none';
}

/**
 * Utility function to remove loading state from element
 * @param {HTMLElement} element - Element to remove loading state from
 */
function removeLoadingState(element) {
    element.classList.remove('loading');
    element.style.opacity = '1';
    element.style.pointerEvents = 'auto';
}

// Export functions if using modules (optional)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        formatTime,
        formatAngle,
        addLoadingState,
        removeLoadingState
    };
}