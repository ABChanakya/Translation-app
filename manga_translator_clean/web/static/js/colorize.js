// Public colorization page functionality.

let uploadedFile = null;
let colorizationAvailable = false;

const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const colorizeBtn = document.getElementById('colorizeBtn');
const intensitySlider = document.getElementById('intensity');
const intensityValue = document.getElementById('intensityValue');
const colorizationStatusChip = document.getElementById('colorizationStatusChip');
const colorizationHint = document.getElementById('colorizationHint');

function updateColorizeButtonState() {
    const statusLocked = colorizeBtn.dataset.disabledByStatus === 'true';
    colorizeBtn.disabled = statusLocked || !uploadedFile || !colorizationAvailable;
}

async function loadColorizationStatus() {
    try {
        const status = await apiRequest('/api/colorize/status');
        colorizationAvailable = status.available;
        colorizationStatusChip.textContent = status.available ? 'Ready' : 'Setup required';
        colorizationStatusChip.classList.toggle('status-chip-warning', !status.available);
        if (!status.available) {
            colorizationHint.textContent = status.enable_instructions;
        }
    } catch (error) {
        colorizationAvailable = false;
        colorizationStatusChip.textContent = 'Status unavailable';
        colorizationStatusChip.classList.add('status-chip-warning');
    } finally {
        updateColorizeButtonState();
    }
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
        updateColorizeButtonState();
    };
    reader.readAsDataURL(file);

    try {
        showNotification('Uploading image...', 'info');
        uploadedFile = await uploadFile(file);
        showNotification('Image uploaded successfully', 'success');
        updateColorizeButtonState();
    } catch (error) {
        clearUpload();
    }
}

function clearUpload() {
    uploadedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    updateColorizeButtonState();
}

async function colorizeImage() {
    if (!uploadedFile) {
        showNotification('Please upload an image first', 'error');
        return;
    }
    if (!colorizationAvailable) {
        showNotification('Colorization backend is not configured yet', 'error');
        return;
    }

    const progressSection = document.getElementById('progressSection');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const resultsSection = document.getElementById('resultsSection');

    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';
    colorizeBtn.disabled = true;
    progressFill.style.width = '10%';

    try {
        progressText.textContent = 'Loading colorization model...';
        const result = await apiRequest('/api/colorize', 'POST', {
            input_path: uploadedFile.filepath,
            model: document.getElementById('colorModel').value,
            intensity: parseInt(document.getElementById('intensity').value, 10)
        });
        progressFill.style.width = '100%';
        progressText.textContent = 'Colorization complete';
        setTimeout(() => {
            progressSection.style.display = 'none';
            showResults(result);
        }, 350);
        showNotification('Colorization completed', 'success');
    } catch (error) {
        progressSection.style.display = 'none';
        showNotification(`Colorization failed: ${error.message}`, 'error');
    } finally {
        updateColorizeButtonState();
    }
}

function showResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    document.getElementById('originalImage').src = previewImage.src;
    document.getElementById('colorizedImage').src = result.output_url;
    resultsSection.style.display = 'block';
}

function downloadResult() {
    const colorizedImage = document.getElementById('colorizedImage');
    const link = document.createElement('a');
    link.href = colorizedImage.src;
    link.download = 'colorized_manga.png';
    link.click();
}

intensitySlider.addEventListener('input', (event) => {
    intensityValue.textContent = `${event.target.value}%`;
});

fileInput.addEventListener('change', handleFileSelect);

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

document.addEventListener('DOMContentLoaded', () => {
    loadColorizationStatus();
    updateColorizeButtonState();
});
