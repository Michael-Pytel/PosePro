Dropzone.autoDiscover = false;

const startButton = document.getElementById("start");
const progressWrapper = document.getElementById("progressWrapper");
const progressBar = document.getElementById("dropzoneProgress");
const statusMessage = document.getElementById("statusMessage");
const dropzoneArea = document.getElementById("dropzoneArea");
const loadingModal = document.getElementById("loadingModal");
const pipelineProgressModal = document.getElementById("pipelineProgressModal");
const currentStageTextModal = document.getElementById("currentStageTextModal");

let activeTimeouts = [];
let activeVideoElement = null;
let progressCheckInterval = null;
let currentProgressPercentage = 0;

// Stage display names
const stageDisplayNames = {
    'segmentation': 'Segmenting Repetitions',
    'landmarks': 'Extracting Pose Landmarks',
    'angles': 'Calculating Joint Angles',
    'aggregation': 'Aggregating Features',
    'prediction': 'Making Predictions'
};

function updateStageUI(stage) {
    // Update the pipeline UI to show the current stage
    const stages = document.querySelectorAll('.pipeline-stage-modal');
    const stageOrder = ['segmentation', 'landmarks', 'angles', 'aggregation', 'prediction'];
    
    stages.forEach((el, index) => {
        const stageAttr = el.getAttribute('data-stage');
        const stageIdx = stageOrder.indexOf(stageAttr);
        const currentIdx = stageOrder.indexOf(stage);
        
        el.classList.remove('active', 'completed');
        
        if (stageIdx < currentIdx) {
            el.classList.add('completed');
        } else if (stageIdx === currentIdx) {
            el.classList.add('active');
            updateStageText(stage);
        }
    });
}

function updateStageText(stage) {
    // Update the status text with animation
    const displayName = stageDisplayNames[stage] || 'Processing...';
    // remove any previous animation classes
    currentStageTextModal.classList.remove('updating', 'zoom');

    // set the text immediately
    currentStageTextModal.textContent = displayName;

    // small timeout to allow DOM to register the text change, then trigger animations
    setTimeout(() => {
        // slide animation
        currentStageTextModal.classList.add('updating');

        // trigger zoom animation: remove then re-add to restart
        void currentStageTextModal.offsetWidth;
        currentStageTextModal.classList.add('zoom');
    }, 20);
}

function checkPipelineProgress() {
    // Poll the server for pipeline progress
    fetch('/api/progress/')
        .then(response => response.json())
        .then(data => {
            const currentStage = data.current_stage;
            const progress = data.progress;
            
            if (currentStage !== 'idle') {
                // Update stage UI
                if (currentStage !== 'complete' && currentStage !== 'error') {
                    updateStageUI(currentStage);
                }
                
                // Update progress percentage if we're past file upload
                if (progress > 0) {
                    currentProgressPercentage = progress;
                }
            }
            
            // If processing complete, stop polling
            if (currentStage === 'complete' || currentStage === 'error') {
                clearInterval(progressCheckInterval);
                if (currentStage === 'complete') {
                    updateStageUI('prediction');
                    document.querySelectorAll('.pipeline-stage-modal').forEach(el => {
                        el.classList.add('completed');
                    });
                }
            }
        })
        .catch(error => console.error('Progress check error:', error));
}

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

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

videoDropzone.on("addedfile", function (file) {
    cleanupPreviousVideo();

    dropzoneArea.classList.add("has-file");
    startButton.disabled = false;
    startButton.textContent = "Send and Process Video";

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
    progressBar.style.background = "linear-gradient(90deg, #667eea, #764ba2)";
    statusMessage.style.display = "none";
    statusMessage.className = "status-message";
}

startButton.onclick = () => {
    if (startButton.textContent === "Upload Another Video") {
        removeVideo();
        return;
    }

    progressWrapper.classList.add("active");
    progressBar.style.background = "linear-gradient(90deg, #667eea, #764ba2)";
    progressBar.style.width = "0%";
    startButton.disabled = true;
    videoDropzone.processQueue();
};

videoDropzone.on("totaluploadprogress", (progress) => {
    progressBar.style.width = progress + "%";
});

videoDropzone.on("sending", function() {
    // Start progress checking when file upload begins
    currentProgressPercentage = 0;
    
    // Initialize all stages as inactive
    document.querySelectorAll('.pipeline-stage-modal').forEach(el => {
        el.classList.remove('active', 'completed');
    });

    // Show modal and set initial active stage
    currentStageTextModal.textContent = 'Preparing...';
    setTimeout(() => {
        loadingModal.classList.add("active");
        // mark first stage active immediately for visual feedback
        updateStageUI('segmentation');
        updateStageText('segmentation');
        // Start checking progress every 500ms
        if (progressCheckInterval) clearInterval(progressCheckInterval);
        progressCheckInterval = setInterval(checkPipelineProgress, 500);
    }, 100);
});

videoDropzone.on("success", (file, response) => {
    // Stop progress checking
    if (progressCheckInterval) clearInterval(progressCheckInterval);
    
    progressBar.style.background = "linear-gradient(90deg, #45d96f, #21a655)";
    loadingModal.classList.remove("active");

    if (response.status === "success") {
        // ⭐ REDIRECT TO RESULTS
        window.location.href = response.redirect_url;
    } else {
        statusMessage.textContent = "✗ Processing failed";
        statusMessage.className = "status-message error";
        startButton.disabled = false;
        startButton.textContent = "Try Again";
    }
});

videoDropzone.on("error", (file, errorMessage) => {
    loadingModal.classList.remove("active");
    statusMessage.textContent = "✗ Upload error: " + errorMessage;
    statusMessage.className = "status-message error";
    startButton.disabled = false;
    startButton.textContent = "Try Again";
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