function setup() {
  canvas = createCanvas(windowWidth, windowHeight);
  canvas.parent('sketch-container');

  canvas.elt.style.touchAction = 'none';
  canvas.elt.style.webkitTouchCallout = 'none';
  canvas.elt.style.webkitUserSelect = 'none';
  canvas.elt.style.userSelect = 'none';

  ['touchstart', 'touchmove', 'touchend', 'touchcancel'].forEach((type) => {
    canvas.elt.addEventListener(type, function (event) {
      event.preventDefault();
      event.stopPropagation();
      if (type === 'touchstart') isDragging = true;
      if (type === 'touchend' || type === 'touchcancel') isDragging = false;
      if (type === 'touchmove' && isDragging) {
        centeredText.classList.add('fade-out');
      }
    }, { passive: false });
  });

  background(0);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  background(0); // ripristina lo sfondo nero dopo il resize
}

function draw() {

  fill(0);
  stroke('#e48b58');
  strokeWeight(2)

  if (mouseIsPressed == true) {
    circle(mouseX, mouseY, 40);
  }
}

// evita di scrollare mentre disegni
function touchStarted() {
  return false;
}

function touchMoved() {
  if (touches.length > 0) {
    fill(0);
    stroke('#e48b58');
    strokeWeight(2);
    circle(touches[0].x, touches[0].y, 40);
  }
  return false;
}

const centeredText = document.querySelector('.centered-text');
const sketchContainer = document.getElementById('sketch-container');


let isDragging = false;

// Detect when the mouse is down (start of drag)
sketchContainer.addEventListener('mousedown', function () {
  isDragging = true;
});

// Detect when the mouse is moving while dragging
sketchContainer.addEventListener('mousemove', function (event) {
  if (isDragging) {
    // When dragging starts, fade out the text
    centeredText.classList.add('fade-out');
  }
});

// Detect when the mouse is released (end of drag)
sketchContainer.addEventListener('mouseup', function () {
  isDragging = false;
});

// For touch devices (optional)
sketchContainer.addEventListener('touchstart', function () {
  isDragging = true;
});

sketchContainer.addEventListener('touchmove', function (event) {
  event.preventDefault();
  if (isDragging) {
    centeredText.classList.add('fade-out');
  }
}, { passive: false });

sketchContainer.addEventListener('touchend', function () {
  isDragging = false;
});

window.addEventListener('DOMContentLoaded', () => {
  const nav = document.querySelector('nav');
  const contactSection = document.querySelector('#contact');

  if (!nav || !contactSection) return;

  const observer = new IntersectionObserver((entries) => {
    const entry = entries[0];
    nav.classList.toggle('hidden', entry.isIntersecting);
  }, { threshold: 0.25 });

  observer.observe(contactSection);
});
