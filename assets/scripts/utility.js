console.log('utility.js loaded');

// Initialize global variables
let drawWidth = 40;
let eraseWidth = 60;
let isDrawing = false;
let isErasing = false;
let lastX = 0;
let lastY = 0;
let ctx = null;
let rect = null;

// Function to start drawing
function startDrawing(e) {
    isDrawing = true;

    [lastX, lastY] = [e.clientX - rect.left, e.clientY - rect.top];

    console.log('startDrawing', lastX, lastY);
}

// Function to draw on the canvas
function draw(e) {
    if (!isDrawing) return;

    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();

    [lastX, lastY] = [currentX, currentY];
}

// Function to stop drawing
function stopDrawing() {
    isDrawing = false;
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

function setupCanvasContext(canvas) {
    ctx = canvas.getContext('2d');
    image = document.getElementById('image');
    props = getImageSize(image);
    ctx.canvas.width = props.width;
    ctx.canvas.height = props.height;

    isDrawing = false;
    isErasing = false;

    console.log('canvas_draw', image.clientWidth, image.clientHeight);

    // Set canvas properties
    ctx.strokeStyle = 'red';
    ctx.lineWidth = drawWidth;
    ctx.lineCap = 'round';

    // Set up mousemove eventlistener that does not need to go back to the app
    canvas.addEventListener('mousemove', draw);
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
            setupCanvasContext(canvas);

            image = document.getElementById('image');
            props = getImageSize(image);

            const img = new Image();
            img.src = data;
            img.onload = function () {
                console.log('Image loaded and drawn on canvas');
                ctx.drawImage(img, 0, 0, props.width, props.height);
            };

            console.log('Waiting for image to load');

            return '';
        },
        canvas_clear: function () {
            if (ctx === null) {
                console.log('No context found');
                return '';
            }
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            // we also have to reset the ctx as a new image might be loaded
            ctx = null;
            return '';
        },
        canvas_toggle_erase: function () {
            className = 'bg-blue-500 text-white p-2 rounded-md';
            if (ctx === null) {
                isErasing = false;
                console.log('No context found');
                return className;
            }
            isErasing = !isErasing;
            if (isErasing) {
                className = 'bg-red-500 text-white p-2 rounded-md';
                ctx.globalCompositeOperation = 'destination-out';
                ctx.strokeStyle = 'rgba(0,0,0,1)';
                ctx.lineWidth = eraseWidth;
                ctx.lineCap = 'round';
            } else {
                ctx.globalCompositeOperation = 'source-over';
                ctx.strokeStyle = 'red';
                ctx.lineWidth = drawWidth;
                ctx.lineCap = 'round';
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
            rect = canvas.getBoundingClientRect();
            if (ctx === null) {
                setupCanvasContext(canvas);
            }

            switch (event.type) {
                case 'mousedown':
                case 'touchstart':
                    startDrawing(event);
                    break;
                case 'mouseup':
                case 'mouseout':
                case 'touchend':
                    stopDrawing(canvas);
                    break;
            }
            return '';
        }
    }
});