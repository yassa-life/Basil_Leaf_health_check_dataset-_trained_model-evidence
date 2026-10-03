"""Local Basil Lab website. Uses the trusted model produced by the notebook."""
from pathlib import Path
from io import BytesIO
import csv
import json
import time

import joblib
from flask import Flask, jsonify, render_template, request, send_file, abort
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
DATA = ROOT / "data" / "raw"
FEATURE_VERSION = "rgb-hsv-lbp-hog-v1"
Image.MAX_IMAGE_PIXELS = 25_000_000
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=12 * 1024 * 1024, TRUSTED_HOSTS=["127.0.0.1", "localhost"])


from parts._pipeline import read_rgb, handcrafted


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
    print("Basil Lab is ready at http://127.0.0.1:8001", flush=True)
    serve(app, host="127.0.0.1", port=8001, threads=4)
