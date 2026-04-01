// Public translation page functionality.

let uploadedFile = null;
let batchFiles = [];
let currentMode = 'single';
let viewMode = 'comparison';
let progressEventSource = null;

const fileInput = document.getElementById('fileInput');
const batchFileInput = document.getElementById('batchFileInput');
const uploadBox = document.getElementById('uploadBox');
const batchUploadBox = document.getElementById('batchUploadBox');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const translateBtn = document.getElementById('translateBtn');
const translationContainer = document.querySelector('.translation-container');
const translatorSelect = document.getElementById('translator');

function generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

async function loadTranslatorEngines() {
    const engineStatusChip = document.getElementById('engineStatusChip');

    try {
        const result = await apiRequest('/api/engines');
        translatorSelect.innerHTML = '';

        if (!result.engines || result.engines.length === 0) {
            translatorSelect.innerHTML = '<option value="">No public engines available</option>';
            translatorSelect.disabled = true;
            engineStatusChip.textContent = 'No public engines available';
            engineStatusChip.classList.add('status-chip-warning');
            updateTranslateAvailability();
            return;
        }

        result.engines.forEach((engine) => {
            const option = document.createElement('option');
            option.value = engine.engine_id;
            option.textContent = engine.label;
            if (engine.engine_id === result.default_engine) {
                option.selected = true;
            }
            translatorSelect.appendChild(option);
        });

        engineStatusChip.textContent = `${result.engines.length} engine(s) available`;
        engineStatusChip.classList.remove('status-chip-warning');
        translatorSelect.disabled = false;
    } catch (error) {
        translatorSelect.innerHTML = '<option value="">Engine list unavailable</option>';
        translatorSelect.disabled = true;
        engineStatusChip.textContent = 'Could not load engines';
        engineStatusChip.classList.add('status-chip-warning');
    } finally {
        updateTranslateAvailability();
    }
}

function updateTranslateAvailability() {
    const hasInput = currentMode === 'single' ? Boolean(uploadedFile) : batchFiles.length > 0;
    translateBtn.disabled = translatorSelect.disabled || !hasInput;
}

function initStoryContextToggle() {
    const toggle = document.getElementById('storyContextToggle');
    const body = document.getElementById('storyContextBody');
    const chevron = document.getElementById('storyContextChevron');
    const badge = document.getElementById('storyContextBadge');
    const textarea = document.getElementById('storyContext');

    if (!toggle || !body || !textarea) return;

    toggle.addEventListener('click', () => {
        const expanded = toggle.getAttribute('aria-expanded') === 'true';
        toggle.setAttribute('aria-expanded', !expanded);
        body.hidden = expanded;
        chevron.style.transform = expanded ? '' : 'rotate(180deg)';
        if (!expanded) textarea.focus();
    });

    textarea.addEventListener('input', () => {
        badge.style.display = textarea.value.trim() ? 'inline' : 'none';
    });
}

function switchMode(mode) {
    currentMode = mode;
    const singleModeBtn = document.getElementById('singleModeBtn');
    const batchModeBtn = document.getElementById('batchModeBtn');
    const singleMode = document.getElementById('singleMode');
    const batchMode = document.getElementById('batchMode');
    const batchOptions = document.getElementById('batchOptions');
    const translateBtnText = document.getElementById('translateBtnText');

    if (mode === 'single') {
        singleModeBtn.classList.remove('btn-outline-primary');
        singleModeBtn.classList.add('btn-primary');
        batchModeBtn.classList.remove('btn-primary');
        batchModeBtn.classList.add('btn-outline-primary');
        singleMode.style.display = 'block';
        batchMode.style.display = 'none';
        batchOptions.style.display = 'none';
        translateBtnText.textContent = 'Translate';
        translateBtn.onclick = translateImage;
    } else {
        batchModeBtn.classList.remove('btn-outline-primary');
        batchModeBtn.classList.add('btn-primary');
        singleModeBtn.classList.remove('btn-primary');
        singleModeBtn.classList.add('btn-outline-primary');
        singleMode.style.display = 'none';
        batchMode.style.display = 'block';
        batchOptions.style.display = 'block';
        translateBtnText.textContent = 'Translate Batch';
        translateBtn.onclick = translateBatch;
    }

    clearUpload();
    clearBatchUpload();
}

function initializePageMode() {
    const initialMode = translationContainer?.dataset.initialMode || 'single';
    switchMode(initialMode === 'batch' ? 'batch' : 'single');
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

async function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('Please select an image file', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        previewImage.src = event.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        updateTranslateAvailability();
    };
    reader.readAsDataURL(file);

    try {
        showNotification('Uploading image...', 'info');
        uploadedFile = await uploadFile(file);
        showNotification('Image uploaded successfully', 'success');
    } catch (error) {
        clearUpload();
    }
}

function clearUpload() {
    uploadedFile = null;
    if (fileInput) {
        fileInput.value = '';
    }
    previewImage.src = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    updateTranslateAvailability();
    document.getElementById('resultsSection').style.display = 'none';
}

async function handleBatchFiles(files) {
    if (files.length === 0) {
        return;
    }
    if (files.length > 100) {
        showNotification('Maximum 100 files allowed per batch request', 'error');
        return;
    }
    for (const file of files) {
        if (!file.type.startsWith('image/')) {
            showNotification('All files must be images', 'error');
            return;
        }
    }

    try {
        showNotification(`Uploading ${files.length} files...`, 'info');
        const formData = new FormData();
        files.forEach((file) => formData.append('files[]', file));

        const response = await fetch('/api/batch/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || 'Batch upload failed');
        }

        batchFiles = result.files;
        displayBatchFiles();
        updateTranslateAvailability();
        showNotification(`${result.count} files uploaded successfully`, 'success');
    } catch (error) {
        showNotification(`Upload failed: ${error.message}`, 'error');
        clearBatchUpload();
    }
}

function handleBatchFileSelect(event) {
    handleBatchFiles(Array.from(event.target.files));
}

function displayBatchFiles() {
    const batchPreviewSection = document.getElementById('batchPreviewSection');
    const batchFileList = document.getElementById('batchFileList');
    const batchFileCount = document.getElementById('batchFileCount');

    batchFileCount.textContent = batchFiles.length;
    batchFileList.innerHTML = '';

    batchFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'batch-file-item';
        fileItem.innerHTML = `
            <i class="fas fa-file-image"></i>
            <span>${file.filename}</span>
            <button class="btn btn-sm btn-danger" onclick="removeBatchFile(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        batchFileList.appendChild(fileItem);
    });

    batchUploadBox.style.display = 'none';
    batchPreviewSection.style.display = 'block';
}

function removeBatchFile(index) {
    batchFiles.splice(index, 1);
    if (batchFiles.length === 0) {
        clearBatchUpload();
    } else {
        displayBatchFiles();
    }
}

function clearBatchUpload() {
    batchFiles = [];
    if (batchFileInput) {
        batchFileInput.value = '';
    }
    document.getElementById('batchUploadBox').style.display = 'block';
    document.getElementById('batchPreviewSection').style.display = 'none';
    updateTranslateAvailability();
}

async function translateImage() {
    if (!uploadedFile) {
        showNotification('Please upload an image first', 'error');
        return;
    }

    const translator = document.getElementById('translator').value;
    const targetLang = document.getElementById('targetLang').value;
    const confidence = parseFloat(document.getElementById('confidence').value);
    const iouThreshold = parseFloat(document.getElementById('iouThreshold').value);
    const storyContext = (document.getElementById('storyContext') || {}).value?.trim() || null;
    const sessionId = generateSessionId();

    const progressSection = document.getElementById('progressSection');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const resultsSection = document.getElementById('resultsSection');

    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';
    translateBtn.disabled = true;
    progressFill.style.width = '0%';
    progressText.textContent = 'Initializing...';

    progressEventSource = new EventSource(`/api/progress/${sessionId}`);
    progressEventSource.onmessage = (event) => {
        const update = JSON.parse(event.data);
        progressFill.style.width = `${update.progress}%`;
        updateStageIndicators(update.stage);
        progressText.textContent = update.message;
        if (update.stage === 'complete' || update.stage === 'error') {
            progressEventSource.close();
            progressEventSource = null;
        }
    };
    progressEventSource.onerror = () => {
        if (progressEventSource) {
            progressEventSource.close();
            progressEventSource = null;
        }
    };

    try {
        const result = await apiRequest('/api/translate', 'POST', {
            input_path: uploadedFile.filepath,
            target_lang: targetLang,
            translator: translator,
            confidence: confidence,
            iou_threshold: iouThreshold,
            story_context: storyContext,
            session_id: sessionId
        });

        await new Promise((resolve) => setTimeout(resolve, 350));
        progressSection.style.display = 'none';
        showResults(result);
        showNotification('Translation completed', 'success');
    } catch (error) {
        if (progressEventSource) {
            progressEventSource.close();
            progressEventSource = null;
        }
        progressSection.style.display = 'none';
    } finally {
        translateBtn.disabled = false;
    }
}

function showResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const originalImage = document.getElementById('originalImage');
    const translatedImage = document.getElementById('translatedImage');
    const comparisonOriginal = document.getElementById('comparisonOriginal');
    const comparisonTranslated = document.getElementById('comparisonTranslated');
    const statsBox = document.getElementById('statsBox');

    originalImage.src = previewImage.src;
    translatedImage.src = result.output_url;
    comparisonOriginal.src = previewImage.src;
    comparisonTranslated.src = result.output_url;

    if (result.stats) {
        let regionDetails = '';
        if (result.stats.regions_by_type) {
            regionDetails = '<ul style="margin: 10px 0; padding-left: 20px;">';
            for (const [typeName, count] of Object.entries(result.stats.regions_by_type)) {
                regionDetails += `<li>${typeName}: ${count}</li>`;
            }
            regionDetails += '</ul>';
        }

        statsBox.innerHTML = `
            <h4>Detection Results</h4>
            <p><strong>Total Regions Detected:</strong> ${result.stats.bubbles_detected || 0}</p>
            ${regionDetails}
            <p class="helper-text">
                Settings used: Confidence=${result.stats.confidence || result.confidence || 'N/A'}, IoU=${result.stats.iou_threshold || result.iou_threshold || 'N/A'}
            </p>
            <div class="stats-actions">
                <button class="btn btn-primary" onclick="downloadResult()">
                    <i class="fas fa-download"></i> Download Translated
                </button>
            </div>
        `;
    }

    document.querySelector('.view-mode-toggle').style.display = 'block';
    switchViewMode('comparison');
    resultsSection.style.display = 'block';
}

function switchViewMode(mode) {
    viewMode = mode;
    const comparisonView = document.getElementById('comparisonView');
    const sideBySideView = document.getElementById('sideBySideView');
    const viewComparisonBtn = document.getElementById('viewComparisonBtn');
    const viewSideBySideBtn = document.getElementById('viewSideBySideBtn');

    if (mode === 'comparison') {
        comparisonView.style.display = 'block';
        sideBySideView.style.display = 'none';
        viewComparisonBtn.classList.remove('btn-outline-primary');
        viewComparisonBtn.classList.add('btn-primary');
        viewSideBySideBtn.classList.remove('btn-primary');
        viewSideBySideBtn.classList.add('btn-outline-primary');
    } else {
        comparisonView.style.display = 'none';
        sideBySideView.style.display = 'grid';
        viewSideBySideBtn.classList.remove('btn-outline-primary');
        viewSideBySideBtn.classList.add('btn-primary');
        viewComparisonBtn.classList.remove('btn-primary');
        viewComparisonBtn.classList.add('btn-outline-primary');
    }
}

function downloadResult() {
    const translatedImage = document.getElementById('translatedImage');
    const link = document.createElement('a');
    link.href = translatedImage.src;
    link.download = 'translated_manga.png';
    link.click();
}

async function translateBatch() {
    if (batchFiles.length === 0) {
        showNotification('Please upload images first', 'error');
        return;
    }

    const translator = document.getElementById('translator').value;
    const targetLang = document.getElementById('targetLang').value;
    const confidence = parseFloat(document.getElementById('confidence').value);
    const iouThreshold = parseFloat(document.getElementById('iouThreshold').value);
    const outputFormat = document.getElementById('outputFormat').value;
    const includeOriginals = document.getElementById('includeOriginals').checked;
    const chunkSize = parseInt(document.getElementById('chunkSize').value, 10);
    const storyContext = (document.getElementById('storyContext') || {}).value?.trim() || null;

    const progressSection = document.getElementById('progressSection');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const resultsSection = document.getElementById('resultsSection');

    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';
    translateBtn.disabled = true;
    progressFill.style.width = '10%';
    progressText.textContent = `Processing ${batchFiles.length} pages in chunks of ${chunkSize}...`;

    try {
        const filePaths = batchFiles.map((file) => file.filepath);
        const response = await fetch('/api/batch/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_paths: filePaths,
                target_lang: targetLang,
                translator: translator,
                confidence: confidence,
                iou_threshold: iouThreshold,
                output_format: outputFormat,
                include_originals: includeOriginals,
                chunk_size: chunkSize,
                story_context: storyContext
            })
        });

        progressFill.style.width = '90%';
        const result = await response.json();
        if (!response.ok) {
            const detail = result.page_errors && result.page_errors.length > 0
                ? ` (first error: ${result.page_errors[0].error})`
                : '';
            throw new Error((result.error || 'Batch translation failed') + detail);
        }

        progressFill.style.width = '100%';
        progressText.textContent = 'Batch complete';
        setTimeout(() => {
            progressSection.style.display = 'none';
            showBatchResults(result);
        }, 350);
        showNotification(`Batch complete: ${result.processed}/${result.total} pages translated`, 'success');
    } catch (error) {
        progressSection.style.display = 'none';
        showNotification(`Batch translation failed: ${error.message}`, 'error');
    } finally {
        translateBtn.disabled = false;
    }
}

function showBatchResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const statsBox = document.getElementById('statsBox');
    const comparisonView = document.getElementById('comparisonView');
    const sideBySideView = document.getElementById('sideBySideView');
    const viewModeToggle = document.querySelector('.view-mode-toggle');

    comparisonView.style.display = 'none';
    sideBySideView.style.display = 'none';
    viewModeToggle.style.display = 'none';

    let downloadsHtml = '<h4>Download Results</h4><div class="batch-downloads">';
    if (result.outputs.zip && result.outputs.zip.url) {
        downloadsHtml += `
            <a href="${result.outputs.zip.url}" class="btn btn-primary" download>
                <i class="fas fa-file-archive"></i> Download ZIP
            </a>
        `;
    } else if (result.outputs.zip && result.outputs.zip.error) {
        downloadsHtml += `<p class="helper-text">ZIP: ${result.outputs.zip.error}</p>`;
    }
    if (result.outputs.pdf && result.outputs.pdf.url) {
        downloadsHtml += `
            <a href="${result.outputs.pdf.url}" class="btn btn-success" download>
                <i class="fas fa-file-pdf"></i> Download PDF
            </a>
        `;
    } else if (result.outputs.pdf && result.outputs.pdf.error) {
        downloadsHtml += `<p class="helper-text">PDF: ${result.outputs.pdf.error}</p>`;
    }
    downloadsHtml += '</div>';

    let errorsHtml = '';
    if (result.errors && result.errors.length > 0) {
        errorsHtml = `
            <div class="alert alert-warning batch-errors">
                <h4>Warnings (${result.errors.length})</h4>
                <ul>
                    ${result.errors.map((error) => `<li>Page ${error.page + 1}: ${error.error}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    statsBox.innerHTML = `
        <h4>Batch Processing Results</h4>
        <p><strong>Total Pages:</strong> ${result.total}</p>
        <p><strong>Successfully Processed:</strong> ${result.processed}</p>
        <p><strong>Failed:</strong> ${result.failed}</p>
        <p><strong>Chunk Size:</strong> ${result.chunk_size}</p>
        ${downloadsHtml}
        ${errorsHtml}
    `;
    resultsSection.style.display = 'block';
}

function updateStageIndicators(currentStage) {
    const stages = ['uploading', 'detecting', 'ocr', 'translating', 'inpainting', 'rendering'];
    const currentIndex = stages.indexOf(currentStage);
    stages.forEach((stage, index) => {
        const element = document.getElementById(`stage-${stage}`);
        if (!element) {
            return;
        }
        element.classList.remove('active', 'complete');
        if (index < currentIndex) {
            element.classList.add('complete');
        } else if (index === currentIndex) {
            element.classList.add('active');
        }
    });
    if (currentStage === 'complete') {
        stages.forEach((stage) => {
            const element = document.getElementById(`stage-${stage}`);
            if (element) {
                element.classList.remove('active');
                element.classList.add('complete');
            }
        });
    }
}

document.getElementById('confidence').addEventListener('input', (event) => {
    document.getElementById('confidenceValue').textContent = event.target.value;
});

document.getElementById('iouThreshold').addEventListener('input', (event) => {
    document.getElementById('iouValue').textContent = event.target.value;
});

fileInput.addEventListener('change', handleFileSelect);
batchFileInput.addEventListener('change', handleBatchFileSelect);

uploadBox.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadBox.classList.add('drag-over');
});
uploadBox.addEventListener('dragleave', (event) => {
    event.preventDefault();
    uploadBox.classList.remove('drag-over');
});
uploadBox.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadBox.classList.remove('drag-over');
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

batchUploadBox.addEventListener('dragover', (event) => {
    event.preventDefault();
    batchUploadBox.classList.add('drag-over');
});
batchUploadBox.addEventListener('dragleave', (event) => {
    event.preventDefault();
    batchUploadBox.classList.remove('drag-over');
});
batchUploadBox.addEventListener('drop', (event) => {
    event.preventDefault();
    batchUploadBox.classList.remove('drag-over');
    handleBatchFiles(Array.from(event.dataTransfer.files));
});

document.addEventListener('DOMContentLoaded', () => {
    loadTranslatorEngines();
    initializePageMode();
    initStoryContextToggle();
});
