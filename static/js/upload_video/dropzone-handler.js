Dropzone.autoDiscover = false;

// DOM Elements
const startButton = document.getElementById("start");
const progressWrapper = document.getElementById("progressWrapper");
const progressBar = document.getElementById("dropzoneProgress");
const statusMessage = document.getElementById("statusMessage");
const dropzoneArea = document.getElementById("dropzoneArea");
const processingModal = document.getElementById("processingModal");
const stageText = document.getElementById("stageText");
const timeElapsed = document.getElementById("timeElapsed");
const processingTip = document.getElementById("processingTip");
const errorModal = document.getElementById("errorModal");
const errorMessage = document.getElementById("errorMessage");
const errorCloseBtn = document.getElementById("errorCloseBtn");

// State
let activeTimeouts = [];
let activeVideoElement = null;
let startTime = null;
let timerInterval = null;
let currentTipIndex = 0;
let progressPollingInterval = null;
let currentSessionId = null;

// Stage to step mapping
const stageToStep = {
    'UPLOADING': 'upload',
    'EXTRACTING_LANDMARKS': 'landmarks',
    'COMPUTING_SIGNALS': 'signals',
    'DETECTING_REPS': 'reps',
    'EXTRACTING_FEATURES': 'features',
    'MAKING_PREDICTIONS': 'prediction',
    'CUTTING_VIDEOS': 'prediction',
    'COMPLETE': 'prediction'
};

// Stage to icon mapping
const stageIcons = {
    'UPLOADING': '',
    'EXTRACTING_LANDMARKS': '',
    'COMPUTING_SIGNALS': '',
    'DETECTING_REPS': '',
    'EXTRACTING_FEATURES': '',
    'MAKING_PREDICTIONS': '',
    'CUTTING_VIDEOS': '',
    'COMPLETE': ''
};

// Tips to display during processing
const processingTips = [
    "Our system analyzes 33 body landmarks to assess your exercise form.",
    "The system evaluates head position, hip alignment, and range of motion.",
    "MediaPipe pose estimation tracks your movement with sub-millimeter precision.",
    "Each repetition is analyzed individually for detailed performance insights.",
    "Biomechanical analysis calculates joint angles to ensure proper form.",
    "The model was trained on thousands of exercise videos for accurate predictions."
];

// Initialize Dropzone
let videoDropzone = new Dropzone("#dropzoneArea", {
    url: '/demo/upload/',
    maxFiles: 1,
    maxFilesize: 150,
    paramName: "video",
    acceptedFiles: ".MP4,.MOV,.AVI,.mp4,.mov,.avi",
    addRemoveLinks: false,
    autoProcessQueue: false,
    createImageThumbnails: false,
    dictDefaultMessage: "Click or drag video here to upload",
    dictFallbackMessage: "Your browser does not support drag and drop file uploads.",
    dictFileTooBig: "File is too big ({{filesize}}MB). Max filesize: {{maxFilesize}}MB.",
    dictInvalidFileType: "Invalid file format. Only MP4, MOV, and AVI files are accepted.",
    dictRemoveFile: "Remove",
    dictCancelUpload: "Cancel upload"
});

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function formatTime(seconds) {
    if (seconds < 60) {
        return `${seconds}s`;
    }
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
}

function updateTimer() {
    if (startTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        timeElapsed.textContent = formatTime(elapsed);
    }
}

function rotateTip() {
    currentTipIndex = (currentTipIndex + 1) % processingTips.length;
    processingTip.style.opacity = '0';
    setTimeout(() => {
        processingTip.textContent = processingTips[currentTipIndex];
        processingTip.style.opacity = '1';
    }, 300);
}

function updateStepIndicators(currentStage) {
    const currentStep = stageToStep[currentStage];
    if (!currentStep) return;
    
    // Get all step items
    const stepItems = document.querySelectorAll('.step-item');
    const stepConnectors = document.querySelectorAll('.step-connector');
    
    // Define step order
    const stepOrder = ['upload', 'landmarks', 'signals', 'reps', 'features', 'prediction'];
    const currentIndex = stepOrder.indexOf(currentStep);
    
    // Update step items
    stepItems.forEach((item, index) => {
        const stepName = item.getAttribute('data-step');
        const stepIndex = stepOrder.indexOf(stepName);
        
        if (stepIndex < currentIndex) {
            // Completed steps
            item.classList.add('completed');
            item.classList.remove('active');
        } else if (stepIndex === currentIndex) {
            // Current step
            item.classList.add('active');
            item.classList.remove('completed');
        } else {
            // Future steps
            item.classList.remove('active', 'completed');
        }
    });
    
    // Update connectors
    stepConnectors.forEach((connector, index) => {
        if (index < currentIndex) {
            connector.classList.add('completed');
        } else {
            connector.classList.remove('completed');
        }
    });
}

function updateProgressUI(progressData) {
    if (!progressData) return;
    
    console.log('Updating UI with progress:', progressData);
    
    // Update stage message
    if (stageText) {
        stageText.textContent = progressData.message;
    }
    
    // Update stage icon
    const stageIcon = document.querySelector('.stage-icon');
    if (stageIcon && stageIcons[progressData.stage]) {
        stageIcon.textContent = stageIcons[progressData.stage];
    }
    
    // Update rep counter (e.g., "Rep 3/10")
    const repCounterContainer = document.getElementById('repCounterContainer');
    const repCounter = document.getElementById('repCounter');
    if (progressData.details && repCounter && repCounterContainer) {
        repCounter.textContent = `Rep ${progressData.details}`;
        repCounterContainer.style.display = 'flex';
    } else if (repCounterContainer) {
        repCounterContainer.style.display = 'none';
    }
    
    // Update step indicators (MAIN VISUAL FEEDBACK)
    updateStepIndicators(progressData.stage);
}

function startProgressPolling(sessionId) {
    console.log('Starting progress polling for session:', sessionId);
    currentSessionId = sessionId;
    
    // Clear any existing polling
    if (progressPollingInterval) {
        clearInterval(progressPollingInterval);
    }
    
    // Poll every 500ms
    progressPollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/progress/?session_id=${sessionId}`);
            
            if (!response.ok) {
                if (response.status === 404) {
                    console.log('Progress data not found yet, continuing to poll...');
                    return;
                }
                console.error('Progress polling error:', response.status);
                return;
            }
            
            const progressData = await response.json();
            console.log('Progress update:', progressData);
            
            updateProgressUI(progressData);
            
            // Stop polling when complete
            if (progressData.progress >= 100 || progressData.stage === 'COMPLETE') {
                console.log('Processing complete, stopping polling');
                stopProgressPolling();
            }
        } catch (error) {
            console.error('Error polling progress:', error);
        }
    }, 500);
}

function stopProgressPolling() {
    if (progressPollingInterval) {
        clearInterval(progressPollingInterval);
        progressPollingInterval = null;
    }
}

function showProcessingModal() {
    console.log('Showing processing modal');
    
    // Reset state
    startTime = Date.now();
    currentTipIndex = 0;
    
    // Show modal
    processingModal.classList.add('active');
    
    // Initialize display
    if (processingTip) {
        processingTip.textContent = processingTips[0];
    }
    
    if (timeElapsed) {
        timeElapsed.textContent = '0s';
    }
    
    if (stageText) {
        stageText.textContent = 'Uploading video...';
    }
    
    // Hide rep counter initially
    const repCounterContainer = document.getElementById('repCounterContainer');
    if (repCounterContainer) {
        repCounterContainer.style.display = 'none';
    }
    
    // Reset step indicators
    document.querySelectorAll('.step-item').forEach(item => {
        item.classList.remove('active', 'completed');
    });
    document.querySelectorAll('.step-connector').forEach(connector => {
        connector.classList.remove('completed');
    });
    
    // Start timer
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(updateTimer, 1000);
    
    // Rotate tips every 8 seconds
    setInterval(rotateTip, 8000);
}

function hideProcessingModal() {
    processingModal.classList.remove('active');
    if (timerInterval) clearInterval(timerInterval);
    stopProgressPolling();
}

// Video Preview Functions
function cleanupPreviousVideo() {
    activeTimeouts.forEach(t => clearTimeout(t));
    activeTimeouts = [];

    if (activeVideoElement) {
        activeVideoElement.pause();

        if (activeVideoElement._playHandler) {
            activeVideoElement.removeEventListener("play", activeVideoElement._playHandler);
        }
        if (activeVideoElement._pauseHandler) {
            activeVideoElement.removeEventListener("pause", activeVideoElement._pauseHandler);
        }

        const src = activeVideoElement.src;
        activeVideoElement.src = '';
        activeVideoElement.load();
        if (src && src.startsWith('blob:')) {
            URL.revokeObjectURL(src);
        }

        activeVideoElement = null;
    }

    const preview = document.getElementById("videoPreviewContainer");
    if (preview) {
        if (preview._clickHandler) {
            preview.removeEventListener("click", preview._clickHandler);
        }
        if (preview.parentNode) {
            preview.remove();
        }
    }
}

function removeVideo() {
    const preview = document.getElementById("videoPreviewContainer");
    if (preview) {
        preview.classList.remove("visible");

        setTimeout(() => {
            cleanupPreviousVideo();
            videoDropzone.removeAllFiles(true);
            dropzoneArea.classList.remove("has-file");
        }, 300);
    }

    startButton.disabled = true;
    startButton.querySelector('span').textContent = "Send and Process Video";
    progressWrapper.classList.remove("active");
    progressBar.style.width = "0%";
    statusMessage.style.display = "none";
    statusMessage.className = "status-message";
}

// Dropzone Event Handlers
videoDropzone.on("addedfile", function (file) {
    console.log('File added:', file.name);

    const allowedFormats = ['mp4', 'mov', 'avi'];
    const fileExtension = file.name.split('.').pop().toLowerCase();

    if (!allowedFormats.includes(fileExtension)) {
        videoDropzone.removeFile(file);
        showErrorModal(`Invalid file format.\n\nOnly MP4, MOV, and AVI files are accepted.\nYour file: ${file.name}`);
        return;
    }

    const maxSize = 150 * 1024 * 1024; // 150 MB
    if (file.size > maxSize) {
        videoDropzone.removeFile(file);
        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        showErrorModal(`File is too large.\n\nMaximum size: 150 MB\nYour file: ${fileSizeMB} MB`);
        return;
    }
    
    cleanupPreviousVideo();

    dropzoneArea.classList.add("has-file");
    startButton.disabled = false;
    startButton.querySelector('span').textContent = "Send and Process Video";

    const wrapper = document.createElement("div");
    wrapper.className = "video-preview-container";
    wrapper.id = "videoPreviewContainer";

    const videoElement = document.createElement("video");
    videoElement.src = URL.createObjectURL(file);
    videoElement.controls = true;
    videoElement.preload = "metadata";

    activeVideoElement = videoElement;

    const overlay = document.createElement("div");
    overlay.className = "video-preview-overlay";

    const fileInfo = document.createElement("div");
    fileInfo.className = "file-info-inline";
    fileInfo.innerHTML = `
        ${file.name}
        <span class="file-size-inline">${formatFileSize(file.size)}</span>
    `;

    overlay.appendChild(fileInfo);

    const playIcon = document.createElement("div");
    playIcon.className = "video-preview-icon";
    playIcon.innerHTML = "▶";

    const removeBtn = document.createElement("button");
    removeBtn.className = "remove-video-btn";
    removeBtn.textContent = "✕ Remove";
    removeBtn.onclick = (e) => {
        e.stopPropagation();
        removeVideo();
    };

    wrapper.appendChild(videoElement);
    wrapper.appendChild(overlay);
    wrapper.appendChild(playIcon);
    wrapper.appendChild(removeBtn);

    dropzoneArea.appendChild(wrapper);

    setTimeout(() => {
        wrapper.classList.add("visible");
    }, 50);

    const wrapperClickHandler = (e) => {
        if (e.target === removeBtn || removeBtn.contains(e.target)) return;
        if (videoElement.paused) {
            videoElement.play();
        } else {
            videoElement.pause();
        }
    };

    const playHandler = () => {
        playIcon.innerHTML = "⏸";
        playIcon.classList.remove("fade-out");

        activeTimeouts.forEach(t => clearTimeout(t));
        activeTimeouts = [];

        const hideTimeout = setTimeout(() => {
            if (videoElement && !videoElement.paused) {
                playIcon.classList.add("fade-out");
            }
        }, 1500);

        activeTimeouts.push(hideTimeout);
    };

    const pauseHandler = () => {
        playIcon.innerHTML = "▶";
        playIcon.classList.remove("fade-out");
        activeTimeouts.forEach(t => clearTimeout(t));
        activeTimeouts = [];
    };

    wrapper.addEventListener("click", wrapperClickHandler);
    videoElement.addEventListener("play", playHandler);
    videoElement.addEventListener("pause", pauseHandler);

    wrapper._clickHandler = wrapperClickHandler;
    videoElement._playHandler = playHandler;
    videoElement._pauseHandler = pauseHandler;

    statusMessage.style.display = "none";
});

startButton.onclick = () => {
    const buttonText = startButton.querySelector('span');
    if (buttonText.textContent === "Upload Another Video") {
        removeVideo();
        return;
    }

    console.log('Starting upload...');
    progressWrapper.classList.add("active");
    progressBar.style.width = "0%";
    startButton.disabled = true;
    videoDropzone.processQueue();
};

videoDropzone.on("totaluploadprogress", (progress) => {
    progressBar.style.width = progress + "%";
});

videoDropzone.on("sending", function(file, xhr, formData) {
    console.log('Sending file...');
    showProcessingModal();
    
    // Generate a session ID for this upload
    currentSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    formData.append('session_id', currentSessionId);
    
    console.log('Generated session ID:', currentSessionId);
});

videoDropzone.on("success", (file, response) => {
    console.log('Upload success:', response);
    
    progressBar.style.background = "linear-gradient(90deg, #45d96f, #21a655)";
    
    // IMMEDIATE: Start polling as soon as upload completes
    if (response.session_id) {
        console.log('Starting progress tracking IMMEDIATELY for session:', response.session_id);
        startProgressPolling(response.session_id);
        
        // Also start checking for completion
        checkForCompletion(response.session_id);
    } else if (currentSessionId) {
        console.log('Using generated session ID:', currentSessionId);
        startProgressPolling(currentSessionId);
        checkForCompletion(currentSessionId);
    }
});

function checkForCompletion(sessionId) {
    const checkInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/check-status/?session_id=${sessionId}`);
            
            if (!response.ok) {
                console.error('Status check error:', response.status);
                return;
            }
            
            const data = await response.json();
            console.log('Status check:', data);
            
            if (data.status === 'complete') {
                clearInterval(checkInterval);
                stopProgressPolling();
                
                // Wait a moment to show completed state, then redirect
                setTimeout(() => {
                    hideProcessingModal();
                    console.log('Redirecting to:', data.redirect_url);
                    window.location.href = data.redirect_url;
                }, 1500);
            } else if (data.status === 'error') {
                clearInterval(checkInterval);
                stopProgressPolling();
                hideProcessingModal();
                showErrorModal(data.error || 'Processing failed');
            }
            // If status is 'processing', keep checking
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }, 2000); // Check every 2 seconds
}

videoDropzone.on("error", (file, errorMessage) => {
    console.error('Upload error:', errorMessage);
    hideProcessingModal();
    
    videoDropzone.removeFile(file);

    let message = "Upload failed. Please try again.";
    if (typeof errorMessage === 'string') {
        message = errorMessage;
    } else if (errorMessage.error) {
        message = errorMessage.error;
    }

    showErrorModal(message);

    startButton.disabled = false;
    startButton.querySelector('span').textContent = "Try Again";
});

videoDropzone.on("queuecomplete", () => {
    if (videoDropzone.getUploadingFiles().length === 0 &&
        videoDropzone.getQueuedFiles().length === 0) {
        setTimeout(() => {
            startButton.disabled = false;
        }, 500);
    }
});

videoDropzone.on("maxfilesexceeded", function(file) {
    videoDropzone.removeFile(file);
});

function showErrorModal(message) {
    errorMessage.textContent = message;
    errorModal.classList.add('active');
}

function hideErrorModal() {
    errorModal.classList.remove('active');
}

if (errorCloseBtn) {
    errorCloseBtn.onclick = hideErrorModal;
}

window.addEventListener("beforeunload", () => {
    cleanupPreviousVideo();
    stopProgressPolling();
});

// Debug logging
console.log('Dropzone handler initialized with progress tracking');
console.log('Processing modal element:', processingModal ? 'Found' : 'NOT FOUND');
console.log('Stage text element:', stageText ? 'Found' : 'NOT FOUND');