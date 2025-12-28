// Results Display JavaScript
// Fetch data from embedded script tag
const resultsData = window.RESULTS_DATA;

if (resultsData) {
    displayResults(resultsData);
} else {
    window.location.href = '/upload/';
}

function displayResults(data) {
    // Display Statistics
    const statsHTML = `
        <div class="stat-card">
            <h4>${data.total_reps || 0}</h4>
            <p>Total Repetitions</p>
        </div>
        <div class="stat-card">
            <h4>${data.repetition_clips ? data.repetition_clips.length : 0}</h4>
            <p>Analyzed Clips</p>
        </div>
        <div class="stat-card">
            <h4>✓</h4>
            <p>Analysis Complete</p>
        </div>
    `;
    document.getElementById("resultsStats").innerHTML = statsHTML;

    // Display Full Video with Visualization
    if (data.visualization_video) {
        const videoHTML = `
            <h3>Full Video Analysis</h3>
            <video controls preload="metadata">
                <source src="${data.visualization_video}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        `;
        document.getElementById("resultsVideoContainer").innerHTML = videoHTML;
    }

    // Display Individual Repetition Clips with Predictions
    if (data.repetition_clips && data.repetition_clips.length > 0) {
        const clipsHTML = data.repetition_clips.map((clip, index) => {
            // Get predictions for this repetition if available
            let predictionsHTML = '';
            if (data.predictions && data.predictions[index]) {
                const pred = data.predictions[index];
                predictionsHTML = `
                    <div class="clip-predictions">
                        <h6>Form Assessment:</h6>
                        <ul>
                            <li>
                                <strong>Head Position:</strong>
                                ${formatPredictionLabel(pred.head_position)}
                            </li>
                            <li>
                                <strong>Hips:</strong>
                                ${formatPredictionLabel(pred.hips)}
                            </li>
                            <li>
                                <strong>Elbows:</strong>
                                ${formatPredictionLabel(pred.elbows)}
                            </li>
                            <li>
                                <strong>Range of Motion:</strong>
                                ${formatPredictionLabel(pred.range_of_motion)}
                            </li>
                        </ul>
                    </div>
                `;
            }
            
            return `
                <div class="clip-card">
                    <video controls preload="metadata">
                        <source src="${clip.path}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <div class="clip-card-info">
                        <h5>Repetition #${index + 1}</h5>
                        <p>${clip.filename}</p>
                        ${predictionsHTML}
                    </div>
                </div>
            `;
        }).join('');
        
        document.getElementById("clipsGrid").innerHTML = clipsHTML;
    }
}

function formatPredictionLabel(value) {
    if (value === 'N/A' || value === null || value === undefined || value === '') {
        return '<span class="prediction-na">N/A</span>';
    }
    
    // Handle numeric values (0/1 or 0.0/1.0)
    const numValue = parseFloat(value);
    
    if (!isNaN(numValue)) {
        if (numValue === 1 || numValue === 1.0) {
            return '<span class="prediction-correct">Correct</span>';
        } else if (numValue === 0 || numValue === 0.0) {
            return '<span class="prediction-incorrect">Incorrect</span>';
        } else {
            // For other numeric values, display with 2 decimal places
            return '<span class="prediction-value">' + numValue.toFixed(2) + '</span>';
        }
    }
    
    // For string values, return as is with value styling
    return '<span class="prediction-value">' + value + '</span>';
}

// Add smooth scroll behavior for video elements
document.addEventListener('DOMContentLoaded', function() {
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
