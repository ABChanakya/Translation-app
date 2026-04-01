// Annotation page functionality

let currentTool = 'box';
let currentClass = 'text';
let annotations = [];
let currentImage = null;
let images = [];
let currentImageIndex = 0;

const canvas = document.getElementById('annotationCanvas');
const ctx = canvas.getContext('2d');

let isDrawing = false;
let startX, startY;

// Tool selection
document.querySelectorAll('.tool-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        currentTool = this.dataset.tool;
    });
});

// Class selection
document.querySelectorAll('.class-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.class-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        currentClass = this.dataset.class;
    });
});

// Canvas mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(e) {
    if (currentTool !== 'box') return;
    
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
}

function draw(e) {
    if (!isDrawing || currentTool !== 'box') return;
    
    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    // Redraw image and existing annotations
    redrawCanvas();
    
    // Draw current box
    ctx.strokeStyle = '#4a90e2';
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
}

function stopDrawing(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    
    // Add annotation
    if (Math.abs(endX - startX) > 10 && Math.abs(endY - startY) > 10) {
        const annotation = {
            class: currentClass,
            x: Math.min(startX, endX),
            y: Math.min(startY, endY),
            width: Math.abs(endX - startX),
            height: Math.abs(endY - startY)
        };
        
        annotations.push(annotation);
        updateAnnotationsList();
        redrawCanvas();
    }
    
    isDrawing = false;
}

function redrawCanvas() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw image
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    }
    
    // Draw annotations
    annotations.forEach((ann, index) => {
        ctx.strokeStyle = getClassColor(ann.class);
        ctx.lineWidth = 2;
        ctx.strokeRect(ann.x, ann.y, ann.width, ann.height);
        
        // Draw label
        ctx.fillStyle = getClassColor(ann.class);
        ctx.fillRect(ann.x, ann.y - 20, 80, 20);
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(ann.class, ann.x + 5, ann.y - 5);
    });
}

function getClassColor(className) {
    const colors = {
        'text': '#4a90e2',
        'sfx': '#50c878',
        'title': '#e74c3c'
    };
    return colors[className] || '#6c757d';
}

function updateAnnotationsList() {
    const list = document.getElementById('annotationsList');
    
    if (annotations.length === 0) {
        list.innerHTML = '<p>No annotations yet</p>';
        return;
    }
    
    list.innerHTML = '';
    
    annotations.forEach((ann, index) => {
        const item = document.createElement('div');
        item.className = 'annotation-item';
        item.innerHTML = `
            <div style="padding: 0.5rem; background: white; border-left: 4px solid ${getClassColor(ann.class)}; margin-bottom: 0.5rem;">
                <strong>${ann.class}</strong><br>
                X: ${Math.round(ann.x)}, Y: ${Math.round(ann.y)}<br>
                W: ${Math.round(ann.width)}, H: ${Math.round(ann.height)}<br>
                <button class="btn btn-secondary" style="font-size: 0.8rem; padding: 0.3rem 0.6rem;" onclick="deleteAnnotation(${index})">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        list.appendChild(item);
    });
}

function deleteAnnotation(index) {
    annotations.splice(index, 1);
    updateAnnotationsList();
    redrawCanvas();
}

function saveAnnotations() {
    if (annotations.length === 0) {
        showNotification('No annotations to save', 'error');
        return;
    }
    
    // Convert to YOLO format
    const yoloAnnotations = annotations.map(ann => {
        const classId = {'text': 0, 'sfx': 1, 'title': 2}[ann.class] || 0;
        const centerX = (ann.x + ann.width / 2) / canvas.width;
        const centerY = (ann.y + ann.height / 2) / canvas.height;
        const width = ann.width / canvas.width;
        const height = ann.height / canvas.height;
        
        return `${classId} ${centerX} ${centerY} ${width} ${height}`;
    }).join('\n');
    
    // Download as .txt file
    const blob = new Blob([yoloAnnotations], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'annotations.txt';
    link.click();
    
    showNotification('Annotations saved!', 'success');
}

function clearAnnotations() {
    if (confirm('Clear all annotations for this image?')) {
        annotations = [];
        updateAnnotationsList();
        redrawCanvas();
    }
}

function exportDataset() {
    showNotification('Dataset export functionality coming soon!', 'info');
}

function prevImage() {
    if (currentImageIndex > 0) {
        currentImageIndex--;
        loadImage(images[currentImageIndex]);
    }
}

function nextImage() {
    if (currentImageIndex < images.length - 1) {
        currentImageIndex++;
        loadImage(images[currentImageIndex]);
    }
}

function loadImage(imagePath) {
    const img = new Image();
    img.onload = function() {
        currentImage = img;
        canvas.width = 800;
        canvas.height = (800 / img.width) * img.height;
        redrawCanvas();
        
        document.getElementById('imageCounter').textContent = 
            `${currentImageIndex + 1} / ${images.length}`;
    };
    img.src = imagePath;
}
