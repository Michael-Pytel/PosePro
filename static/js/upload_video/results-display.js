// ⭐ POBIERZ DANE Z EMBEDDED SCRIPT TAG
const resultsData = window.RESULTS_DATA;

if (resultsData) {
    displayResults(resultsData);
} else {
    window.location.href = '/';
}

function displayResults(data) {
    // Statystyki
    const statsHTML = `
        <div class="stat-card">
            <h4>${data.total_reps}</h4>
            <p>Total Repetitions</p>
        </div>
        <div class="stat-card">
            <h4>${data.repetition_clips.length}</h4>
            <p>Analyzed Clips</p>
        </div>
        <div class="stat-card">
            <h4>✓</h4>
            <p>Processing Complete</p>
        </div>
    `;
    document.getElementById("resultsStats").innerHTML = statsHTML;

    // Główne wideo z wizualizacją
    if (data.visualization_video) {
        const videoHTML = `
            <h3 style="margin-bottom: 15px;">Full Video Analysis</h3>
            <video controls>
                <source src="${data.visualization_video}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        `;
        document.getElementById("resultsVideoContainer").innerHTML = videoHTML;
    }

    // Klipy powtórzeń z predykcjami
    const clipsHTML = data.repetition_clips.map((clip, index) => {
        // Get predictions for this repetition if available
        let predictionsHTML = '';
        if (data.predictions && data.predictions[index]) {
            const pred = data.predictions[index];
            predictionsHTML = `
                <div class="clip-predictions">
                    <h6>Model Predictions:</h6>
                    <ul>
                        <li><strong>Head Position:</strong> ${formatPredictionLabel(pred.head_position)}</li>
                        <li><strong>Hips:</strong> ${formatPredictionLabel(pred.hips)}</li>
                        <li><strong>Elbows:</strong> ${formatPredictionLabel(pred.elbows)}</li>
                        <li><strong>Range of Motion:</strong> ${formatPredictionLabel(pred.range_of_motion)}</li>
                    </ul>
                </div>
            `;
        }
        
        return `
            <div class="clip-card">
                <video controls>
                    <source src="${clip.path}" type="video/mp4">
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

function formatPredictionLabel(value) {
    if (value === 'N/A' || value === null || value === undefined) {
        return '<span class="prediction-na">N/A</span>';
    }
    
    // Handle numeric values (0/1 or 0.0/1.0)
    const numValue = parseFloat(value);
    
    if (numValue === 1 || numValue === 1.0) {
        return '<span class="prediction-correct">✓ Correct</span>';
    } else if (numValue === 0 || numValue === 0.0) {
        return '<span class="prediction-incorrect">✗ Incorrect</span>';
    }
    
    // If it's a different number, format it nicely
    return '<span class="prediction-value">' + numValue.toFixed(2) + '</span>';
}