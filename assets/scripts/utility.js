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

let currentTab = 'Mode'; // Default tab is 'Mode' - should not be hardcoded
let currentSlice = null; // Default slice is null

// For Zooming
let zoomLevel = 1;
let zoomFactor = 1.1;

// For Brush Sizing
let isAltRightDragging = false;
let initialDragX = 0;

// Store last preview coordinates for more efficient erasing
let lastPreviewX = null;
let lastPreviewY = null;
let lastBrushRadius = null;
let gCtx = null;
let gRect = null;

let gPreviewCtx = null;

// setup a resize observer to the canvas and recompute the rect
const resizeObserver = new ResizeObserver(entries => {
    var canvas = entries[0].target;
    gRect = canvas.getBoundingClientRect();
});

function resetContext() {
    gCtx = null;
    gPreviewCtx = null;
    gRect = null;
}

function clearPreviewCanvas() {
    // Clear previous preview (if any)
    if (lastPreviewX !== null && lastPreviewY !== null) {
        const pixelRatio = getPixelRatio(gPreviewCtx);
        const clearX = lastPreviewX - lastBrushRadius - 3 * pixelRatio;
        const clearY = lastPreviewY - lastBrushRadius - 3 * pixelRatio;
        const clearWidth = 2 * lastBrushRadius + 6 * pixelRatio;
        const clearHeight = 2 * lastBrushRadius + 6 * pixelRatio;
        gPreviewCtx.clearRect(clearX, clearY, clearWidth, clearHeight);
    }
}

// Preview the brush size
function previewBrush(e) {
    clearPreviewCanvas();

    const pixelRatio = getPixelRatio(gPreviewCtx);
    const brushSize = isErasing ? eraseWidth : drawWidth;
    const brushRadius = brushSize * pixelRatio / 2;
    const [currentX, currentY] = translateCoordinates(e);

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

// Shows the selected points for segmentation
function previewPoint(e) {
    const [currentX, currentY] = translateCoordinates(e);

    const pixelRatio = getPixelRatio(gPreviewCtx);
    gPreviewCtx.beginPath();
    gPreviewCtx.arc(currentX, currentY, 5 * pixelRatio, 0, 2 * Math.PI);

    color = e.ctrlKey ? 'rgba(255, 0, 0, 1)' : 'rgba(0, 255, 0, 1)';
    gPreviewCtx.fillStyle = color;
    gPreviewCtx.fill();
}

// Shows the bounding box for the selected slice and mask
// for a couple of seconds as a preview and then clears it
function previewRect(ctx, rect) {
    imgElement = document.getElementById('image');
    var scalingFactor = getImageScalingFactor(imgElement);

    const pixelRatio = getPixelRatio(ctx);
    ctx.strokeStyle = 'rgba(255, 128, 0, 1)';
    ctx.lineWidth = 2 * pixelRatio;

    const adjustedRect = [
        rect[0] * scalingFactor,
        rect[1] * scalingFactor,
        rect[2] * scalingFactor,
        rect[3] * scalingFactor
    ];

    const x = adjustedRect[0];
    const y = adjustedRect[1]
    const width = adjustedRect[2] - x;
    const height = adjustedRect[3] - y;

    ctx.strokeRect(x, y, width, height);

    setTimeout(() => {
        ctx.clearRect(
            x - ctx.lineWidth, y - ctx.lineWidth,
            width + ctx.lineWidth * 2, height + ctx.lineWidth * 2);
    }, 2000);

    console.log('Previewing bounding box: {x:', x, ', y:', y, ', width:', width, ', height:', height, '}');
}

// Function to start drawing
function startDrawing(e) {
    if (e.button === 2 && e.altKey) { // Right-click + Alt key
        isAltRightDragging = true;
        initialDragX = e.clientX;
        return;
    }

    isDrawing = true;

    [lastX, lastY] = translateCoordinates(e);

    console.log('startDrawing', lastX, lastY);
}

function adjustBrushSize(deltaX) {
    const newSize = Math.max(5, Math.min(100, drawWidth + deltaX / 15));
    if (!isErasing) {
        drawWidth = newSize;
    } else {
        eraseWidth = newSize;
    }

    const pixelRatio = getPixelRatio(gCtx);
    gCtx.lineWidth = newSize * pixelRatio;
}

// Function to draw on the canvas
function draw(e) {
    if (currentSlice == null) {
        return;
    }

    if (!isDrawing) {
        if (isAltRightDragging) {
            // Adjust size based on drag distance
            adjustBrushSize(e.clientX - initialDragX);
        }
        previewBrush(e);
        return;
    } else {
        clearPreviewCanvas();
    }

    const [currentX, currentY] = translateCoordinates(e);

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
    isAltRightDragging = false;
}

function translateCoordinates(e) {
    var canvas = document.getElementById('canvas');
    const transform = window.getComputedStyle(canvas).transform;
    const matrix = new DOMMatrixReadOnly(transform);
    const invertedMatrix = matrix.inverse();

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const transformedPoint = invertedMatrix.transformPoint({ x: mouseX, y: mouseY });
    const adjustedX = transformedPoint.x;
    const adjustedY = transformedPoint.y;

    return [adjustedX, adjustedY];
}

function handleWheel(e) {
    e.preventDefault();

    if (e.deltaY < 0) {
        zoomLevel *= zoomFactor;
    } else {
        zoomLevel /= zoomFactor;
    }
    zoomLevel = Math.min(Math.max(0.125, zoomLevel), 8); // Limit zoom level

    console.log('Zoom level:', zoomLevel);
    console.log('Offset:', e.offsetX, e.offsetY);

    // Apply zoom transformation to the image container
    image.style.transform = `scale(${zoomLevel})`;
    image.style.transformOrigin = `${e.offsetX}px ${e.offsetY}px`;

    var canvas = document.getElementById('canvas');
    canvas.style.transform = `scale(${zoomLevel})`;
    canvas.style.transformOrigin = `${e.offsetX}px ${e.offsetY}px`;

    var preview = document.getElementById('preview-canvas');
    preview.style.transform = `scale(${zoomLevel})`;
    preview.style.transformOrigin = `${e.offsetX}px ${e.offsetY}px`;

    gRect = canvas.getBoundingClientRect();
    previewBrush(e);
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


function getImageScalingFactor(imgElement) {
    var containerRect = imgElement.getBoundingClientRect();
    var scalingFactor = Math.min(containerRect.width / imgElement.naturalWidth,
        containerRect.height / imgElement.naturalHeight);
    return scalingFactor;
}

// Try to get the image size
function getImageSize(imgElement) {
    var scalingFactor = getImageScalingFactor(imgElement);
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
    canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
    });
    canvas.addEventListener('wheel', handleWheel);

    window.addEventListener('resize', resetContext);

    resizeObserver.observe(canvas);
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
        store_current_tab: function (tab) {
            if (tab == null) {
                console.log('No tab found');
            } else {
                console.log('Current tab:', tab);
                currentTab = tab;
            }
            resetContext();
            return window.dash_clientside.no_update;
        },
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
            if (gPreviewCtx === null) {
                gPreviewCtx = setupCanvasCtx(previewCanvas);
            }

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
        preview_canvas_clear: function () {
            if (gPreviewCtx === null) {
                console.log('No preview context found');
                return '';
            }
            gPreviewCtx.clearRect(0, 0, gPreviewCtx.canvas.width, gPreviewCtx.canvas.height);
            gPreviewCtx = null;
            return '';
        },
        canvas_toggle_erase: function () {
            className = 'general-element';
            if (gCtx === null) {
                isErasing = false;
                console.log('No context found');
                return className;
            }
            pixelRatio = getPixelRatio(gCtx);
            isErasing = !isErasing;
            if (isErasing) {
                className = 'erase-color p-2 rounded-md';
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
            if (gPreviewCtx === null) {
                gPreviewCtx = setupCanvasCtx(previewCanvas);
            }

            // we can't draw if we don't have a slice
            if (currentSlice == null) {
                return window.dash_clientside.no_update;
            }

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
        },
        suppress_contextmenu: function (id) {
            console.log('Suppressing context menu for', id);
            setupHelper(id);
            var element = document.getElementById(id);
            element.addEventListener('contextmenu', (e) => {
                // Check if the ctrl key was pressed
                if (e.ctrlKey) {
                    // Prevent the default action
                    e.preventDefault();

                    // Clone the original event properties
                    const newEvent = cloneEvent(e, 'click');

                    // Dispatch the new event to the original target
                    e.target.dispatchEvent(newEvent);
                }
            });
            return window.dash_clientside.no_update;
        },
        record_selected_slice: function (slice) {
            console.log('Selected slice:', slice);
            currentSlice = slice;
            return slice;
        },
        visualize_point: function (e) {
            console.log('Previewing visualized point:', e);
            var previewCanvas = document.getElementById('preview-canvas');
            if (gPreviewCtx === null) {
                gPreviewCtx = setupCanvasCtx(previewCanvas);
            }

            previewPoint(e);

            return window.dash_clientside.no_update;
        },
        show_bounding_box: function (rect) {
            console.log('Bounding box:', rect);
            if (gPreviewCtx === null) {
                gPreviewCtx = setupCanvasCtx(document.getElementById('preview-canvas'));
            }
            previewRect(gPreviewCtx, rect);
            return window.dash_clientside.no_update;
        }
    }
});

function cloneEvent(e, type) {
    return new MouseEvent(type, {
        bubbles: true,
        cancelable: true,
        composed: true,
        view: e.view,
        detail: e.detail,
        screenX: e.screenX,
        screenY: e.screenY,
        clientX: e.clientX,
        clientY: e.clientY,
        ctrlKey: e.ctrlKey,
        altKey: e.altKey,
        shiftKey: e.shiftKey,
        metaKey: e.metaKey,
        button: e.button,
        relatedTarget: e.relatedTarget
    });
}

function setupHelper(id) {
    console.log('Setting up helper for', id);
    var element = document.getElementById(id);
    if (element === null) {
        console.log('No element found with id', id);
        return;
    }

    // Reference to the help window element
    const helpWindow = document.getElementById('help-window');

    // Set the time (in milliseconds) to wait before showing the help window
    const delay = 3000; // 3 seconds

    let timer;
    let mouseX = 0;
    let mouseY = 0;

    // Function to show the help window
    const showHelpWindow = () => {
        helpWindow.style.left = `${mouseX + 10}px`;
        helpWindow.style.top = `${mouseY + 10}px`;
        helpWindow.classList.remove('hidden');

        // Create a list of different help texts
        const helpTexts = {
            segmentation: [
                'Click on the element in the image to create a mask for it',
                'Control-click to remove the element from the mask',
                'Shift-click to add the element to the mask'
            ],
            inpainting: [
                'Drag Alt + Right-click to adjust brush size',
                'Use the mouse wheel to zoom in and out of the image',
                'You can clean up the image by panting over the areas you want to remove and ' + 
                'pressing the erase button to remove them'
            ],
            export: [
                'Click on the export button to download a glTF scene for Blender',
                'Using mesh displacement will give the scene more 3D depth. Try it out!',
                'Upscaling the textures will allow you to zoom in more on the scene.'
            ],
            configuration: [
                'State is saved automatically in a new folder in the directory you are running ' +
                'Parallax Maker in. Eventually, save state will give you a zip file.',
                'You can restore your state by clicking the load button and ' +
                'navigating to the directory on your local machine where the tool is running.',
            ]
        };

        var image = document.getElementById('image');
        if (image.src == '') {
            var modeSpecificHelpTexts = [
                'Drag an image to the input image box to start the workflow',
                'To start working with Parallax Maker, you need to drag an image to the input box'
            ];
        } else if (currentSlice == null && currentTab == 'Inpainting') {
            var modeSpecificHelpTexts = [
                'Select a slice in the segmentation tab to start working with the inpainting tool',
                'To start the inpainting workflow, you need to select a slice in the segmentation tab'
            ];
        } else {
            // check that the current mode is in the help texts
            if (currentTab.toLowerCase() in helpTexts) {
                var modeSpecificHelpTexts = helpTexts[currentTab.toLowerCase()];
            } else {
                var modeSpecificHelpTexts = [
                    'Watch the Youtube tutorial to learn how to use the tool',
                    'Ask for help on Github by creating an issue in the repository',
                ];
            }
        }

        // Randomly select a help text
        const helpText = modeSpecificHelpTexts[Math.floor(Math.random() * modeSpecificHelpTexts.length)];
        helpWindow.innerHTML = helpText;
    }

    // Function to hide the help window
    const hideHelpWindow = () => {
        helpWindow.classList.add('hidden');
    }

    // Function to reset the timer
    const resetTimer = () => {
        hideHelpWindow();
        clearTimeout(timer);
        timer = setTimeout(showHelpWindow, delay);
    }

    const endTimer = () => {
        clearTimeout(timer);
        hideHelpWindow();
    }

    // Event listener to update mouse coordinates
    element.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        resetTimer();
    });

    // Add event listeners for other actions to reset the timer
    element.addEventListener('mousedown', resetTimer); // For mouse clicks
    element.addEventListener('keypress', resetTimer); // For keyboard events
    element.addEventListener('mouseout', endTimer); // Stop showing help when mouse leaves the element
    element.addEventListener('mouseenter', resetTimer); // Start timer when the mouse enters the element
}