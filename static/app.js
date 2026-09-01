const input = document.querySelector('#image-input');
const drop = document.querySelector('#drop-zone');
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
function setFile(file) {
  if (busy) return;
  currentFile = null; analyze.disabled = true; resetResult(); clearPreview(); error('');
  if (!file) return;
  if (file.size > 12 * 1024 * 1024) { input.value = ''; return error('Choose an image smaller than 12 MB.'); }
  if (!/\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name)) { input.value = ''; return error('Choose a JPG, PNG, WebP, BMP or TIFF image.'); }
  currentFile = file; previewUrl = URL.createObjectURL(file); preview.src = previewUrl;
  previewWrap.hidden = false; document.querySelector('#file-name').textContent = file.name;
  document.querySelector('#file-detail').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ready to analyze`;
  analyze.disabled = false; status.textContent = 'READY TO CHECK';
}
preview.addEventListener('error', () => {
  // TIFF/BMP previews are browser-dependent; the backend may still decode the image.
  preview.hidden = true;
  document.querySelector('#file-detail').textContent = 'Preview unavailable · you can still analyze the image';
});
preview.addEventListener('load', () => { preview.hidden = false; });
input.addEventListener('change', () => setFile(input.files[0]));
document.querySelector('#remove-image').addEventListener('click', () => {
  if (busy) return;
  input.value = ''; currentFile = null; clearPreview(); resetResult(); error(''); analyze.disabled = true;
});
for (const name of ['dragenter', 'dragover']) drop.addEventListener(name, event => { event.preventDefault(); if (!busy) drop.classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) drop.addEventListener(name, event => { event.preventDefault(); drop.classList.remove('dragging'); });
drop.addEventListener('drop', event => {
  if (busy) return;
  if (event.dataTransfer.files.length !== 1) return error('Please choose one leaf photo at a time.');
  input.value = ''; setFile(event.dataTransfer.files[0]);
});

async function runPrediction(url, body) {
  if (busy) return;
  busy = true; error(''); resetResult(); status.textContent = 'ANALYZING…';
  analyze.textContent = 'Analyzing your leaf…'; analyze.disabled = true; input.disabled = true;
  document.querySelector('#remove-image').disabled = true;
  document.querySelectorAll('[data-example]').forEach(button => { button.disabled = true; });
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const response = await fetch(url, {method:'POST', body, signal:controller.signal});
    let data;
    try { data = await response.json(); } catch { throw new Error('The server returned an unexpected response. Please try again.'); }
    if (!response.ok) throw new Error(data.error || 'This image could not be analyzed.');
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
    clearTimeout(timeout); busy = false; input.disabled = false;
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
  currentFile = null; input.value = ''; clearPreview();
  preview.src = `/example/${button.dataset.example}.jpg`; previewWrap.hidden = false;
  document.querySelector('#file-name').textContent = button.textContent.replace('↗','').trim();
  document.querySelector('#file-detail').textContent = 'Existing test-set image · demonstration only';
  runPrediction(`/api/example/${button.dataset.example}`);
}));
