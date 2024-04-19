console.log('utility.js loaded');

// Initialize variables
let isDrawing = false;
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


window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        store_rect_coords: function () {
            const graphElement = document.getElementById('image');
            if (graphElement === null) {
                console.log('No element found with id "image"');
                return '';
            }
            rect = graphElement.getBoundingClientRect();
            //console.log(rect);
            return rect;
        },
        canvas_get: function () {
            canvas = document.getElementById('canvas');
            if (canvas === null) {
                console.log('No element found with id "canvas"');
                return '';
            }
            return canvas.toDataURL('image/png');
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
                ctx = canvas.getContext('2d');
                image = document.getElementById('image');
                props = getImageSize(image);
                ctx.canvas.width = props.width;
                ctx.canvas.height = props.height;

                console.log('canvas_draw', image.clientWidth, image.clientHeight);

                // Set canvas properties
                ctx.strokeStyle = 'red';
                ctx.lineWidth = 15;
                ctx.lineCap = 'round';

                // Set up mousemove eventlistener that does not need to go back to the app
                canvas.addEventListener('mousemove', draw);
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