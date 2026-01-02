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

// State
let activeTimeouts = [];
let activeVideoElement = null;
let startTime = null;
let timerInterval = null;
let currentTipIndex = 0;

// Tips to display during processing
const processingTips = [
    "Our AI analyzes 33 body landmarks in real-time to assess your exercise form.",
    "The system evaluates head position, hip alignment, elbow placement, and range of motion.",
    "We use Random Forest classifiers achieving over 92% accuracy in form assessment.",
    "MediaPipe pose estimation tracks your movement with sub-millimeter precision.",
    "Each repetition is analyzed individually for detailed performance insights.",
    "Biomechanical analysis calculates joint angles to ensure proper form.",
    "The model was trained on thousands of exercise videos for accurate predictions."
];

// Initialize Dropzone
let videoDropzone = new Dropzone("#dropzoneArea", {
    url: '/upload/',
    maxFiles: 1,
    paramName: "video",
    acceptedFiles: "video/*",
    addRemoveLinks: false,
    autoProcessQueue: false,
    createImageThumbnails: false,
    dictDefaultMessage: "Click or drag video here to upload",
    dictFallbackMessage: "Your browser does not support drag and drop file uploads.",
    dictFileTooBig: "File is too big ({{filesize}}MB). Max filesize: {{maxFilesize}}MB.",
    dictInvalidFileType: "You can't upload files of this type.",
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
        stageText.textContent = 'Processing your video...';
    }
    
    // Start timer
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(updateTimer, 1000);
    
    // Rotate tips every 8 seconds
    setInterval(rotateTip, 8000);
}

function hideProcessingModal() {
    processingModal.classList.remove('active');
    if (timerInterval) clearInterval(timerInterval);
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
    startButton.textContent = "Send and Process Video";
    progressWrapper.classList.remove("active");
    progressBar.style.width = "0%";
    statusMessage.style.display = "none";
    statusMessage.className = "status-message";
}

// Dropzone Event Handlers
videoDropzone.on("addedfile", function (file) {
    console.log('File added:', file.name);
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

videoDropzone.on("sending", function() {
    console.log('Sending file...');
    showProcessingModal();
});

videoDropzone.on("success", (file, response) => {
    console.log('Upload success:', response);
    
    progressBar.style.background = "linear-gradient(90deg, #45d96f, #21a655)";
    hideProcessingModal();

    if (response.status === "success") {
        console.log('Redirecting to:', response.redirect_url);
        window.location.href = response.redirect_url;
    } else {
        statusMessage.textContent = "✗ Processing failed";
        statusMessage.className = "status-message error";
        startButton.disabled = false;
        startButton.querySelector('span').textContent = "Try Again";
    }
});

videoDropzone.on("error", (file, errorMessage) => {
    console.error('Upload error:', errorMessage);
    hideProcessingModal();
    
    statusMessage.textContent = "✗ Upload error: " + errorMessage;
    statusMessage.className = "status-message error";
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

window.addEventListener("beforeunload", cleanupPreviousVideo);

// Debug logging
console.log('Dropzone handler initialized');
console.log('Processing modal element:', processingModal ? 'Found' : 'NOT FOUND');
console.log('Stage text element:', stageText ? 'Found' : 'NOT FOUND');