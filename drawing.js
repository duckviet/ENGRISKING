
const LINE_COLOUR = '#FFFFFF';
const LINE_WIDTH = 10;

var currentX = 0;
var currentY = 0;
var previousX = 0;
var previousY = 0;

var canvas;
var context;

function prepareCanvas() {
    canvas = document.getElementById('my-canvas');
    context = canvas.getContext('2d');

    // Set canvas width and height based on CSS properties
    canvas.width = parseInt(getComputedStyle(canvas).width, 10);
    canvas.height = parseInt(getComputedStyle(canvas).height, 10);

    context.strokeStyle = LINE_COLOUR;
    context.lineWidth = LINE_WIDTH;
    context.lineJoin = 'round';

    var isPainting = false;

    function startDrawing(event) {
        isPainting = true;
        currentX = event.clientX - canvas.getBoundingClientRect().left;
        currentY = event.clientY - canvas.getBoundingClientRect().top;
    }

    function continueDrawing(event) {
        if (isPainting) {
            previousX = currentX;
            currentX = event.clientX - canvas.getBoundingClientRect().left;
            previousY = currentY;
            currentY = event.clientY - canvas.getBoundingClientRect().top;
            draw();
        }
    }

    function stopDrawing() {
        isPainting = false;
    }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', continueDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    // Touch Events
    canvas.addEventListener('touchstart', function (event) {
        startDrawing(event.touches[0]);
    });

    canvas.addEventListener('touchmove', function (event) {
        continueDrawing(event.touches[0]);
    });

    canvas.addEventListener('touchend', stopDrawing);
    canvas.addEventListener('touchcancel', stopDrawing);
}

function draw() {
    context.beginPath();
    context.moveTo(previousX, previousY);
    context.lineTo(currentX, currentY);
    context.closePath();
    context.stroke();
}

function clearCanvas() {
    currentX = 0;
    currentY = 0;
    previousX = 0;
    previousY = 0;
    // Clear only the canvas without affecting the CSS background
    context.clearRect(0, 0, canvas.width, canvas.height);
}

prepareCanvas() 

// // Call prepareCanvas() to initialize the canvas
// function captureElementAndDrawOnCanvas(elementId, canvasId) {
//     // Get the HTML element to capture
//     const elementToCapture = document.getElementById(elementId);

//     // Create an img element to store the captured image
//     const imgElement = document.createElement('img');

//     // Capture the element's content as a base64-encoded image
//     const base64Image = getBase64Image(elementToCapture);

//     // Set the src attribute of the img element to the captured image
//     imgElement.src = base64Image;

//     // Get the canvas element
//     const canvas = document.getElementById(canvasId);
//     const context = canvas.getContext('2d');

//     // Draw the captured image onto the canvas
//     imgElement.onload = function () {
//         context.drawImage(imgElement, 0, 0);
//     };
// }

// function getBase64Image(element) {
//     // Create a new canvas element to temporarily draw the content
//     const canvas = document.createElement('canvas');
//     const context = canvas.getContext('2d');

//     // Set the canvas dimensions to match the element
//     canvas.width = element.clientWidth;
//     canvas.height = element.clientHeight;

//     // Draw the element's content onto the canvas
//     context.drawImage(element, 0, 0);

//     // Return the base64-encoded image data
//     return canvas.toDataURL('image/png');
// }
