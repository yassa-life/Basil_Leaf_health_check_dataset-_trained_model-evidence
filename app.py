"""Local Basil Lab website. Uses the trusted model produced by the notebook."""
from pathlib import Path
from io import BytesIO
import csv
import json
import time
import warnings

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, abort
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
DATA = ROOT / "data" / "raw"
FEATURE_VERSION = "rgb-hsv-lbp-hog-v1"
Image.MAX_IMAGE_PIXELS = 25_000_000
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=12 * 1024 * 1024, TRUSTED_HOSTS=["127.0.0.1", "localhost"])


def read_rgb(source):
    """Same decoding and feature computation as the self-contained notebook."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(source) as opened:
            if opened.format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                raise ValueError("Use a JPEG, PNG, WebP, BMP or TIFF image.")
            if min(opened.size) < 16:
                raise ValueError("Image is too small; both dimensions must be at least 16 pixels.")
            opened.load()
            return ImageOps.exif_transpose(opened).convert("RGB")


def handcrafted(image):
    # Keep identical to the notebook; tested against its actual function.
    rgb = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    hsv = rgb2hsv(rgb)
    parts = []
    for array in (rgb, hsv):
        for channel in range(3):
            h, _ = np.histogram(array[:, :, channel], bins=16, range=(0, 1))
            parts.append(h.astype(np.float32) / h.sum())
        parts.extend([array.mean(axis=(0, 1)), array.std(axis=(0, 1))])
    gray = (rgb2gray(rgb) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    h, _ = np.histogram(lbp, bins=np.arange(11), density=False)
    parts.append(h.astype(np.float32) / h.sum())
    small = np.asarray(image.resize((64, 64)).convert("L"), dtype=np.float32) / 255
    parts.append(hog(small, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
    return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)


# Only our own fixed local artifact is loaded. Uploaded files are never deserialized.
bundle = joblib.load(OUTPUT / "model.joblib")
report = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
if bundle["feature_version"] != FEATURE_VERSION or bundle["feature"] != "handcrafted":
    raise RuntimeError("The saved model needs a different feature pipeline. Update the website before serving it.")
if bundle["dataset_fingerprint"] != report["dataset_fingerprint"] or bundle["name"] != report["selected_model"]:
    raise RuntimeError("The model and report do not match. Complete notebook training, then restart the website.")
with (OUTPUT / "split_counts.csv").open(newline="", encoding="utf-8") as file:
    split_counts = list(csv.DictReader(file))
with (OUTPUT / "cv_fold_counts.csv").open(newline="", encoding="utf-8") as file:
    folds = list(csv.DictReader(file))
with (OUTPUT / "test_predictions.csv").open(newline="", encoding="utf-8") as file:
    test_rows = list(csv.DictReader(file))
examples = []
for label in bundle["classes"]:
    row = next(row for row in test_rows if row["label"] == label)
    path = (DATA / row["path"]).resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise RuntimeError("Invalid example path in the saved manifest.")
    examples.append({"label": label, "path": path, "filename": path.name})
report["split_counts"] = split_counts
report["cv_folds"] = folds
report["model_parameters"] = json.loads(json.dumps(bundle["estimator"].get_params(deep=False), default=str))
report["seed"] = bundle["seed"]
report["feature_count"] = int(bundle["estimator"].n_features_in_)
report["environment"] = json.loads((OUTPUT / "environment.json").read_text())


@app.before_request
def local_origin_only():
    if request.method == "POST" and request.headers.get("Origin") not in {None, request.host_url.rstrip("/")}:
        return jsonify(error="Please use the upload form on this local website."), 403


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return render_template("index.html", r=report, m=report["metrics"], audit=report["data_audit"],
                           coverage=report["data_audit"]["download_coverage"], examples=examples)


@app.get("/api/report")
def training_report():
    return jsonify(report)


def prediction(source, example=None):
    start = time.perf_counter()
    im = read_rgb(source)
    label = str(bundle["estimator"].predict(handcrafted(im)[None, :])[0])
    return {"label": label, "model": bundle["name"], "width": im.width, "height": im.height,
            "elapsed_ms": round((time.perf_counter() - start) * 1000), "example_label": example,
            "notice": "Experimental quality classification. Not a disease diagnosis or food-safety assessment. Unrelated images cannot be reliably rejected."}


@app.post("/api/predict")
def predict():
    image = request.files.get("image")
    if image is None or not image.filename:
        return jsonify(error="Choose a leaf photo first."), 400
    try:
        # The upload is decoded in memory and is never saved to disk or used for training.
        return jsonify(prediction(image.stream))
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        return jsonify(error="We couldn't read this image. Use an undamaged JPEG, PNG, WebP, BMP or TIFF under 12 MB and 25 megapixels."), 400


@app.post("/api/example/<int:example_id>")
def predict_example(example_id):
    if example_id not in range(len(examples)):
        abort(404)
    example = examples[example_id]
    return jsonify(prediction(example["path"], example["label"]))


@app.get("/example/<int:example_id>.jpg")
def example_image(example_id):
    if example_id not in range(len(examples)):
        abort(404)
    im = read_rgb(examples[example_id]["path"])
    im.thumbnail((1000, 1000))
    buffer = BytesIO()
    im.save(buffer, "JPEG", quality=88)
    buffer.seek(0)
    return send_file(buffer, mimetype="image/jpeg", max_age=3600)


@app.get("/download/<name>")
def download(name):
    files = {"notebook": ROOT / "Basil_Leaf_ML_Workflow.ipynb", "report": OUTPUT / "results.json",
             "comparison": OUTPUT / "model_comparison.csv", "splits": OUTPUT / "split_manifest.csv"}
    if name not in files:
        abort(404)
    return send_file(files[name], as_attachment=True)


@app.errorhandler(413)
def too_large(error):
    return jsonify(error="This file is too large. Choose an image smaller than 12 MB."), 413


if __name__ == "__main__":
    from waitress import serve
    print("Basil Lab is ready at http://127.0.0.1:8000", flush=True)
    serve(app, host="127.0.0.1", port=8000, threads=4)
