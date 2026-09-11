const input = document.querySelector('#image-input');
const cameraInput = document.querySelector('#camera-input');
const drop = document.querySelector('#drop-zone');
const pasteButton = document.querySelector('#paste-image');
const cameraButton = document.querySelector('#take-photo');
const inputHelp = document.querySelector('#input-help');
const analyze = document.querySelector('#analyze');
const previewWrap = document.querySelector('#preview-wrap');
const preview = document.querySelector('#preview');
const resultCard = document.querySelector('#result-card');
const status = document.querySelector('#result-status');
const empty = document.querySelector('#empty-result');
const result = document.querySelector('#prediction-result');
const errorBox = document.querySelector('#upload-error');
let currentFile = null;
let previewUrl = null;
let busy = false;
const allowedTypes = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/x-ms-bmp', 'image/tiff', 'image/tif']);

function error(message) { errorBox.textContent = message; errorBox.hidden = !message; }
function resetResult() {
  result.hidden = true; empty.hidden = false;
  resultCard.classList.remove('unhealthy');
  status.textContent = 'AWAITING PHOTO';
}
function clearPreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null; preview.removeAttribute('src'); previewWrap.hidden = true;
}
function setFile(file, source = 'file upload') {
  if (busy) return;
  currentFile = null; analyze.disabled = true; resetResult(); clearPreview(); error('');
  if (!file) return;
  if (file.size > 12 * 1024 * 1024) { input.value = ''; cameraInput.value = ''; return error('Choose an image smaller than 12 MB.'); }
  const allowedExtension = /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name || '');
  const unsupportedImageType = file.type.startsWith('image/') && !allowedTypes.has(file.type);
  if (unsupportedImageType || (!allowedTypes.has(file.type) && !allowedExtension)) { input.value = ''; cameraInput.value = ''; return error('Choose a JPG, PNG, WebP, BMP or TIFF image.'); }
  currentFile = file; previewUrl = URL.createObjectURL(file); preview.src = previewUrl;
  previewWrap.hidden = false; document.querySelector('#file-name').textContent = file.name;
  document.querySelector('#file-detail').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ${source} · ready to analyze`;
  analyze.disabled = false; status.textContent = 'READY TO CHECK';
  inputHelp.textContent = 'Image ready. Analyze it now or replace it using any input option.';
}
preview.addEventListener('error', () => {
  // TIFF/BMP previews are browser-dependent; the backend may still decode the image.
  preview.hidden = true;
  document.querySelector('#file-detail').textContent = 'Preview unavailable · you can still analyze the image';
});
preview.addEventListener('load', () => { preview.hidden = false; });
input.addEventListener('change', () => setFile(input.files[0], 'file upload'));
cameraInput.addEventListener('change', () => setFile(cameraInput.files[0], 'camera capture'));
cameraButton.addEventListener('click', () => {
  if (busy) return;
  cameraInput.value = '';
  cameraInput.click();
});

function clipboardImage(items) {
  for (const item of items) {
    if (item.kind === 'file' && item.type.startsWith('image/')) return item.getAsFile();
  }
  return null;
}

function namedClipboardFile(blob) {
  const extensions = {'image/jpeg':'jpg', 'image/png':'png', 'image/webp':'webp', 'image/bmp':'bmp', 'image/x-ms-bmp':'bmp', 'image/tiff':'tiff', 'image/tif':'tif'};
  const extension = extensions[blob.type] || 'unsupported';
  return new File([blob], `pasted-basil-leaf.${extension}`, {type: blob.type || 'application/octet-stream'});
}

document.addEventListener('paste', event => {
  if (busy) return;
  const pasted = clipboardImage(event.clipboardData?.items || []);
  if (!pasted) return error('The clipboard does not contain an image. Copy a JPG, PNG, WebP, BMP or TIFF image and try again.');
  event.preventDefault();
  input.value = ''; cameraInput.value = '';
  setFile(namedClipboardFile(pasted), 'pasted image');
});

pasteButton.addEventListener('click', async () => {
  if (busy) return;
  if (!navigator.clipboard?.read) {
    inputHelp.textContent = 'Copy an image, return to this page, and press Ctrl+V to paste it.';
    return error('Direct clipboard access is unavailable in this browser. Use Ctrl+V after copying an image.');
  }
  try {
    const clipboardItems = await navigator.clipboard.read();
    const item = clipboardItems.find(entry => entry.types.some(type => type.startsWith('image/')));
    if (!item) return error('The clipboard does not contain an image. Copy an image and try again.');
    const type = item.types.find(value => value.startsWith('image/'));
    const blob = await item.getType(type);
    input.value = ''; cameraInput.value = '';
    setFile(namedClipboardFile(blob), 'pasted image');
  } catch (err) {
    inputHelp.textContent = 'Clipboard permission was not available. Copy an image and press Ctrl+V on this page.';
    error(err.name === 'NotAllowedError'
      ? 'Clipboard access was blocked. Copy an image, return here, and press Ctrl+V.'
      : 'The clipboard image could not be read. Copy it again or use file upload.');
  }
});
document.querySelector('#remove-image').addEventListener('click', () => {
  if (busy) return;
  input.value = ''; cameraInput.value = ''; currentFile = null; clearPreview(); resetResult(); error(''); analyze.disabled = true;
  inputHelp.textContent = 'Choose one clear basil leaf photo. A new selection replaces the current preview.';
});
for (const name of ['dragenter', 'dragover']) drop.addEventListener(name, event => { event.preventDefault(); if (!busy) drop.classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) drop.addEventListener(name, event => { event.preventDefault(); drop.classList.remove('dragging'); });
drop.addEventListener('drop', event => {
  if (busy) return;
  if (event.dataTransfer.files.length !== 1) return error('Please choose one leaf photo at a time.');
  input.value = ''; cameraInput.value = ''; setFile(event.dataTransfer.files[0], 'drag and drop');
});

async function runPrediction(url, body) {
  if (busy) return;
  busy = true; error(''); resetResult(); status.textContent = 'ANALYZING…';
  analyze.textContent = 'Analyzing your leaf…'; analyze.disabled = true; input.disabled = true; cameraInput.disabled = true;
  pasteButton.disabled = true; cameraButton.disabled = true;
  document.querySelector('#remove-image').disabled = true;
  document.querySelectorAll('[data-example]').forEach(button => { button.disabled = true; });
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const response = await fetch(url, {method:'POST', body, signal:controller.signal});
    let data;
    try { data = await response.json(); } catch { throw new Error('The server returned an unexpected response. Please try again.'); }
    if (!response.ok) {
      const exampleRouteMissing = response.status === 404 && url.startsWith('/api/example/');
      throw new Error(exampleRouteMissing
        ? 'The dataset-example endpoint is unavailable. Start this Basil Lab project and open http://127.0.0.1:8001.'
        : (data.error || 'This image could not be analyzed.'));
    }
    empty.hidden = true; result.hidden = false; status.textContent = 'ANALYSIS COMPLETE';
    resultCard.classList.toggle('unhealthy', data.label === 'Unhealthy');
    document.querySelector('#prediction-label').textContent = data.label;
    document.querySelector('#prediction-description').textContent = data.label === 'Healthy'
      ? 'The image’s visual patterns match the healthy-labelled basil images learned by the model. This does not rule out a leaf-health problem.'
      : 'The image’s visual patterns match the unhealthy-labelled basil images learned by the model. Inspect the leaf; this prediction does not identify a cause.';
    document.querySelector('#prediction-model').textContent = data.model;
    document.querySelector('#prediction-size').textContent = `${data.width} × ${data.height} px`;
    document.querySelector('#prediction-time').textContent = `${data.elapsed_ms} ms · decoding + features + model`;
    const note = document.querySelector('#example-notice');
    note.hidden = !data.example_label;
    note.textContent = data.example_label ? `Existing test-set example · dataset label: ${data.example_label}. This demonstration is not new validation.` : '';
  } catch (err) {
    resetResult(); status.textContent = 'CHECK NOT COMPLETED';
    error(err.name === 'AbortError' ? 'This took too long. Try a smaller image or restart the local website.' : (err.message === 'Failed to fetch' ? 'Cannot reach the local model. Restart the website using start_web.bat.' : err.message));
  } finally {
    clearTimeout(timeout); busy = false; input.disabled = false; cameraInput.disabled = false;
    pasteButton.disabled = false; cameraButton.disabled = false;
    analyze.innerHTML = 'Analyze leaf <span aria-hidden="true">↗</span>'; analyze.disabled = !currentFile;
    document.querySelector('#remove-image').disabled = false;
    document.querySelectorAll('[data-example]').forEach(button => { button.disabled = false; });
  }
}
analyze.addEventListener('click', () => {
  if (!currentFile || busy) return;
  const form = new FormData(); form.append('image', currentFile);
  runPrediction('/api/predict', form);
});
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  if (busy) return;
  currentFile = null; input.value = ''; cameraInput.value = ''; clearPreview();
  preview.src = `/example/${button.dataset.example}.jpg`; previewWrap.hidden = false;
  document.querySelector('#file-name').textContent = button.textContent.replace('↗','').trim();
  document.querySelector('#file-detail').textContent = 'Existing test-set image · demonstration only';
  runPrediction(`/api/example/${button.dataset.example}`);
}));
