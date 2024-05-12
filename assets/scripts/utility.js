console.log('utility.js loaded');

// Initialize global variables
let canvasLastDrawnTime = 0;
let canvasLastSavedTime = 0;
let drawWidth = 40;
let eraseWidth = 60;
let isDrawing = false;
let isErasing = false;
let lastX = 0;
let lastY = 0;
// Store last preview coordinates for more efficient erasing
let lastPreviewX = null;
let lastPreviewY = null;
let lastBrushRadius = null;
let gCtx = null;
let gRect = null;

let gPreviewCtx = null;

function clearPreviewCanvas() {
    // Clear previous preview (if any)
    if (lastPreviewX !== null && lastPreviewY !== null) {
        const clearX = lastPreviewX - lastBrushRadius - 5;
        const clearY = lastPreviewY - lastBrushRadius - 5;
        const clearWidth = 2 * lastBrushRadius + 10;
        const clearHeight = 2 * lastBrushRadius + 10;
        gPreviewCtx.clearRect(clearX, clearY, clearWidth, clearHeight);
    }
}

// Preview the brush size
function previewBrush(e) {
    clearPreviewCanvas();

    const pixelRatio = getPixelRatio(gPreviewCtx);
    const brushSize = isErasing ? eraseWidth : drawWidth;
    const brushRadius = brushSize * pixelRatio / 2;
    const currentX = e.clientX - gRect.left;
    const currentY = e.clientY - gRect.top;

    gPreviewCtx.beginPath();
    gPreviewCtx.arc(currentX, currentY, brushRadius, 0, 2 * Math.PI);
    gPreviewCtx.strokeStyle = isErasing ? 'rgba(0, 0, 0, 0.5)' : 'rgba(255, 0, 0, 0.5)'; // Semi-transparent color
    gPreviewCtx.lineWidth = 2 * pixelRatio; // Thin outline
    gPreviewCtx.stroke();

    // Update last preview coordinates
    lastPreviewX = currentX;
    lastPreviewY = currentY;
    lastBrushRadius = brushRadius;
}


// Function to start drawing
function startDrawing(e) {
    isDrawing = true;

    [lastX, lastY] = [e.clientX - gRect.left, e.clientY - gRect.top];

    console.log('startDrawing', lastX, lastY);
}

// Function to draw on the canvas
function draw(e) {
    if (!isDrawing) {
        previewBrush(e);
        return;
    } else {
        clearPreviewCanvas();
    }

    const currentX = e.clientX - gRect.left;
    const currentY = e.clientY - gRect.top;

    gCtx.beginPath();
    gCtx.moveTo(lastX, lastY);
    gCtx.lineTo(currentX, currentY);
    gCtx.stroke();

    canvasLastDrawnTime = Date.now();

    [lastX, lastY] = [currentX, currentY];
}

// Function to stop drawing
function stopDrawing() {
    isDrawing = false;
}

function getPixelRatio(context) {
  dpr = window.devicePixelRatio || 1,
    bsr = context.webkitBackingStorePixelRatio ||
    context.mozBackingStorePixelRatio ||
    context.msBackingStorePixelRatio ||
    context.oBackingStorePixelRatio ||
    context.backingStorePixelRatio || 1;

  return dpr / bsr;
}

// Try to get the image size
function getImageSize(imgElement) {
    var containerRect = imgElement.getBoundingClientRect();
    var scalingFactor = Math.min(containerRect.width / imgElement.naturalWidth,
        containerRect.height / imgElement.naturalHeight);
    var renderedWidth = imgElement.naturalWidth * scalingFactor;
    var renderedHeight = imgElement.naturalHeight * scalingFactor;

    console.log("Rendered Width:", renderedWidth);
    console.log("Rendered Height:", renderedHeight);

    return { width: renderedWidth, height: renderedHeight };
};

function setupCanvasCtx(canvas) {
    var tmpCtx = canvas.getContext('2d');
    var image = document.getElementById('image');
    var props = getImageSize(image);
    tmpCtx.canvas.width = props.width;
    tmpCtx.canvas.height = props.height;
    return tmpCtx;
}

function setupMainCanvas(canvas) {
    gCtx = setupCanvasCtx(canvas);

    isDrawing = false;
    isErasing = false;

    pixelRatio = getPixelRatio(gCtx);

    console.log('Pixel ratio:', pixelRatio);

    // Set canvas properties
    gCtx.strokeStyle = 'red';
    gCtx.lineWidth = drawWidth * pixelRatio;
    gCtx.lineCap = 'round';

    // Set up mousemove eventlistener that does not need to go back to the app
    canvas.addEventListener('mousemove', drawCallback);
}

function drawCallback(e) {
    requestAnimationFrame(() => draw(e));
}

function resolveRect(graphElement, resolve) {
    const rect = graphElement.getBoundingClientRect();
    const rendered = getImageSize(graphElement);
    rect['width'] = rendered['width'];
    rect['height'] = rendered['height'];
    resolve(rect);
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        store_rect_coords: function () {
            return new Promise((resolve, reject) => {
                const graphElement = document.getElementById('image');
                if (graphElement === null) {
                    console.log('No element found with id "image"');
                    resolve({ x: 0, y: 0, width: 0, height: 0 });
                    return;
                }

                // Check if the image has already loaded
                if (graphElement.complete && graphElement.naturalWidth !== 0) {
                    resolveRect(graphElement, resolve);
                } else {
                    // If the image hasn't loaded yet, wait for the load event
                    graphElement.addEventListener('load', () => {
                        resolveRect(graphElement, resolve);
                    });
                }
            });
        },
        canvas_get: function () {
            canvas = document.getElementById('canvas');
            if (canvas === null) {
                console.log('No element found with id "canvas"');
                return '';
            }
            return canvas.toDataURL('image/png');
        },
        canvas_load: function (data) {
            canvas = document.getElementById('canvas');
            if (canvas === null) {
                console.log('No element found with id "canvas"');
                return '';
            }

            gRect = canvas.getBoundingClientRect();
            if (gCtx === null) {
                setupMainCanvas(canvas);
            }

            var previewCanvas = document.getElementById('preview-canvas');
            gPreviewCtx = setupCanvasCtx(previewCanvas);

            image = document.getElementById('image');
            props = getImageSize(image);

            const img = new Image();
            img.src = data;
            img.onload = function () {
                console.log('Image loaded and drawn on canvas');
                gCtx.drawImage(img, 0, 0, props.width, props.height);
            };

            console.log('Waiting for image to load');

            return '';
        },
        canvas_clear: function () {
            if (gCtx === null) {
                console.log('No context found');
                return '';
            }
            canvasLastSavedTime = Date.now();
            gCtx.clearRect(0, 0, gCtx.canvas.width, gCtx.canvas.height);
            // we also have to reset the ctx as a new image might be loaded
            gCtx = null;
            return '';
        },
        canvas_toggle_erase: function () {
            className = 'bg-blue-500 text-white p-2 rounded-md';
            if (gCtx === null) {
                isErasing = false;
                console.log('No context found');
                return className;
            }
            pixelRatio = getPixelRatio(gCtx);
            isErasing = !isErasing;
            if (isErasing) {
                className = 'bg-red-500 text-white p-2 rounded-md';
                gCtx.globalCompositeOperation = 'destination-out';
                gCtx.strokeStyle = 'rgba(0,0,0,1)';
                gCtx.lineWidth = eraseWidth * pixelRatio;
                gCtx.lineCap = 'round';
            } else {
                gCtx.globalCompositeOperation = 'source-over';
                gCtx.strokeStyle = 'red';
                gCtx.lineWidth = drawWidth * pixelRatio;
                gCtx.lineCap = 'round';
            }
            console.log('isErasing', isErasing);

            return className;
        },
        canvas_draw: function (event) {
            if (event === null) {
                console.log('No event found');
                return '';
            }
            canvas = document.getElementById('canvas');
            if (canvas === null) {
                console.log('No element found with id "canvas"');
                return '';
            }
            gRect = canvas.getBoundingClientRect();
            if (gCtx === null) {
                setupMainCanvas(canvas);
            }

            var previewCanvas = document.getElementById('preview-canvas');
            gPreviewCtx = setupCanvasCtx(previewCanvas);

            switch (event.type) {
                case 'mousedown':
                case 'touchstart':
                    startDrawing(event);
                    break;
                case 'mouseout':
                    // Try to get the canvas without needing to save it
                    stopDrawing(canvas);

                    clearPreviewCanvas();

                    bShouldSave = canvasLastDrawnTime > canvasLastSavedTime;
                    canvasLastSavedTime = Date.now();
                    return bShouldSave ? this.canvas_get() : window.dash_clientside.no_update;
                case 'mouseup':
                case 'touchend':
                    stopDrawing(canvas);
                    break;
                case 'mouseenter':
                    // Do nothing; we just want to set up the canvas
                    console.log('Mouse entered; setup canvas');
                    break;
            }
            return window.dash_clientside.no_update;
        }
    }
});