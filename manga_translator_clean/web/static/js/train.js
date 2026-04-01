// Training page functionality

let trainingInProgress = false;
let lossChart = null;
let mapChart = null;

// Load datasets on page load
document.addEventListener('DOMContentLoaded', () => {
    loadDatasets();
    loadModels();
});

async function loadDatasets() {
    try {
        const result = await apiRequest('/api/datasets');
        const datasetSelect = document.getElementById('dataset');
        
        datasetSelect.innerHTML = '<option value="">Select dataset...</option>';
        
        result.datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.path;
            option.textContent = dataset.name;
            datasetSelect.appendChild(option);
        });
    } catch (error) {
        showNotification('Failed to load datasets', 'error');
    }
}

async function loadModels() {
    try {
        const result = await apiRequest('/api/models');
        const modelsList = document.getElementById('modelsList');
        
        if (result.models.length === 0) {
            modelsList.innerHTML = '<p>No models available yet. Train your first model!</p>';
            return;
        }
        
        modelsList.innerHTML = '';
        
        result.models.forEach(model => {
            const modelCard = document.createElement('div');
            modelCard.className = 'model-card';
            modelCard.innerHTML = `
                <h4>${model.name}</h4>
                <p>Size: ${(model.size / 1024 / 1024).toFixed(2)} MB</p>
                <button class="btn btn-primary" onclick="downloadModel('${model.path}')">
                    <i class="fas fa-download"></i> Download
                </button>
            `;
            modelCard.style.cssText = `
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            `;
            modelsList.appendChild(modelCard);
        });
    } catch (error) {
        showNotification('Failed to load models', 'error');
    }
}

async function startTraining() {
    if (trainingInProgress) {
        showNotification('Training already in progress', 'error');
        return;
    }
    
    const modelSize = document.getElementById('modelSize').value;
    const dataset = document.getElementById('dataset').value;
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    const imageSize = parseInt(document.getElementById('imageSize').value);
    const patience = parseInt(document.getElementById('patience').value);
    const useAugmentation = document.getElementById('useAugmentation').checked;
    const usePretrained = document.getElementById('usePretrained').checked;
    
    if (!dataset) {
        showNotification('Please select a dataset', 'error');
        return;
    }
    
    trainingInProgress = true;
    const trainBtn = document.getElementById('trainBtn');
    trainBtn.disabled = true;
    trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
    
    const trainingStatus = document.getElementById('trainingStatus');
    trainingStatus.innerHTML = `
        <h4>Training Configuration</h4>
        <p>Model: ${modelSize}</p>
        <p>Dataset: ${dataset}</p>
        <p>Epochs: ${epochs}</p>
        <p>Batch Size: ${batchSize}</p>
        <p>Image Size: ${imageSize}</p>
        <p>Status: <strong>Training in progress...</strong></p>
    `;
    
    // Initialize charts
    initializeCharts();
    
    // Simulate training (replace with actual API call)
    try {
        showNotification('Training started!', 'success');
        
        // TODO: Replace with actual API call to start training
        // const result = await apiRequest('/api/train', 'POST', {
        //     model: modelSize,
        //     dataset: dataset,
        //     epochs: epochs,
        //     batch_size: batchSize,
        //     image_size: imageSize,
        //     patience: patience,
        //     augmentation: useAugmentation,
        //     pretrained: usePretrained
        // });
        
        // Simulate training progress
        simulateTraining(epochs);
        
    } catch (error) {
        showNotification('Failed to start training: ' + error.message, 'error');
        trainingInProgress = false;
        trainBtn.disabled = false;
        trainBtn.innerHTML = '<i class="fas fa-play"></i> Start Training';
    }
}

function initializeCharts() {
    const metricsContainer = document.getElementById('metricsContainer');
    metricsContainer.style.display = 'block';
    
    // Loss chart
    const lossCtx = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#4a90e2',
                tension: 0.4
            }, {
                label: 'Validation Loss',
                data: [],
                borderColor: '#e74c3c',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    // mAP chart
    const mapCtx = document.getElementById('mapChart').getContext('2d');
    mapChart = new Chart(mapCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'mAP@0.5',
                data: [],
                borderColor: '#50c878',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

function simulateTraining(totalEpochs) {
    let currentEpoch = 0;
    
    const interval = setInterval(() => {
        currentEpoch++;
        
        // Simulate metrics
        const trainLoss = 1.5 * Math.exp(-currentEpoch / 20) + Math.random() * 0.1;
        const valLoss = 1.6 * Math.exp(-currentEpoch / 20) + Math.random() * 0.15;
        const map = (1 - Math.exp(-currentEpoch / 15)) * 0.9 + Math.random() * 0.05;
        
        // Update charts
        lossChart.data.labels.push(currentEpoch);
        lossChart.data.datasets[0].data.push(trainLoss);
        lossChart.data.datasets[1].data.push(valLoss);
        lossChart.update();
        
        mapChart.data.labels.push(currentEpoch);
        mapChart.data.datasets[0].data.push(map);
        mapChart.update();
        
        // Update status
        const trainingStatus = document.getElementById('trainingStatus');
        trainingStatus.innerHTML = `
            <h4>Training Progress</h4>
            <p>Epoch: ${currentEpoch} / ${totalEpochs}</p>
            <p>Train Loss: ${trainLoss.toFixed(4)}</p>
            <p>Val Loss: ${valLoss.toFixed(4)}</p>
            <p>mAP@0.5: ${map.toFixed(4)}</p>
            <p>Status: <strong>Training...</strong></p>
        `;
        
        if (currentEpoch >= totalEpochs) {
            clearInterval(interval);
            finishTraining();
        }
    }, 1000);
}

function finishTraining() {
    trainingInProgress = false;
    const trainBtn = document.getElementById('trainBtn');
    trainBtn.disabled = false;
    trainBtn.innerHTML = '<i class="fas fa-play"></i> Start Training';
    
    const trainingStatus = document.getElementById('trainingStatus');
    trainingStatus.innerHTML += '<p style="color: #50c878; font-weight: bold;">Training completed!</p>';
    
    showNotification('Training completed successfully!', 'success');
    
    // Reload models list
    loadModels();
}

function downloadModel(path) {
    window.location.href = path;
}
