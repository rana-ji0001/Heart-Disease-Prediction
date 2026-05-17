# app.py

from flask import Flask, render_template, request
from dotenv import load_dotenv
from google import genai
from google.genai import types
import joblib
import pandas as pd
import os
import json
import time
import random
import re

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env file")

client = genai.Client(api_key=api_key)

artifact = joblib.load("model.pkl")
models = artifact["models"]
scaler = artifact["scaler"]
FEATURE_NAMES = artifact["features"]
accuracies = artifact["accuracies"]

GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]


def is_retryable_gemini_error(error):
    text = str(error).lower()
    retry_words = [
        "429",
        "503",
        "quota",
        "rate limit",
        "resource_exhausted",
        "unavailable",
        "temporarily",
        "overloaded",
    ]
    return any(word in text for word in retry_words)


def call_gemini_with_retry(make_request, retries=5):
    last_error = None

    for attempt in range(retries):
        try:
            return make_request()
        except Exception as e:
            last_error = e

            if not is_retryable_gemini_error(e):
                raise e

            wait_time = min(60, (2 ** attempt) * 4) + random.uniform(0, 2)
            print(f"Gemini busy or rate-limited. Retry {attempt + 1}/{retries} in {wait_time:.1f}s")
            time.sleep(wait_time)

    raise ValueError(f"Gemini failed after retries: {last_error}")


def clean_json_response(result_text):
    result_text = result_text.strip()

    if "```" in result_text:
        parts = result_text.split("```")
        if len(parts) >= 2:
            result_text = parts[1].strip()
            if result_text.lower().startswith("json"):
                result_text = result_text[4:].strip()

    try:
        return json.loads(result_text)
    except Exception:
        match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if match:
            return json.loads(match.group())

    raise ValueError(f"JSON parse failed: {result_text}")


def extract_features_from_report(report_text=None, file_bytes=None, mime_type=None):
    MAX_CHARS = 12000

    if report_text and len(report_text) > MAX_CHARS:
        report_text = report_text[:MAX_CHARS]

    prompt = f"""
Extract heart disease prediction features from this medical report.

Return ONLY one valid JSON object with exactly these keys:
{FEATURE_NAMES}

Rules:
- No explanation
- No markdown
- No extra text
- Use numbers only
- If a value is missing, use 0
- oldpeak can be decimal
"""

    contents = [prompt]

    if file_bytes and mime_type:
        contents.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
    elif report_text:
        contents.append(f"Patient report:\n{report_text}")
    else:
        raise ValueError("No report text or file data provided")

    def make_request():
        last_error = None

        for model_name in GEMINI_MODELS:
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
            except Exception as e:
                last_error = e
                print(f"{model_name} failed: {e}")

                if is_retryable_gemini_error(e):
                    raise e

        raise ValueError(f"All Gemini models failed: {last_error}")

    response = call_gemini_with_retry(make_request)
    result_text = response.text.strip()
    features = clean_json_response(result_text)

    if isinstance(features, list):
        features = features[0]

    final_features = {}
    for f in FEATURE_NAMES:
        val = features.get(f, 0)
        try:
            if f == "oldpeak":
                final_features[f] = float(val)
            else:
                final_features[f] = int(float(val))
        except Exception:
            final_features[f] = 0.0 if f == "oldpeak" else 0

    time.sleep(5)
    return final_features


MAPPINGS = {
    "sex": {0: "Female", 1: "Male"},
    "fbs": {0: "No", 1: "Yes"},
    "exang": {0: "No", 1: "Yes"},
    "cp": {
        0: "Asymptomatic",
        1: "Atypical Angina",
        2: "Non-Anginal Pain",
        3: "Typical Angina",
    },
    "restecg": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "LV Hypertrophy",
    },
    "slope": {
        0: "Downsloping",
        1: "Flat",
        2: "Upsloping",
    },
    "thal": {
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect",
    },
}

DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Gender",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Results",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia",
}


def coerce_features(features):
    coerced = {}

    for f in FEATURE_NAMES:
        raw = features.get(f, None)

        if raw is None or raw == "":
            coerced[f] = 0.0 if f == "oldpeak" else 0
            continue

        try:
            if f == "oldpeak":
                coerced[f] = float(raw)
            else:
                coerced[f] = int(float(raw))
        except Exception:
            coerced[f] = 0.0 if f == "oldpeak" else 0

    return coerced


def make_prediction(inputs):
    X = pd.DataFrame([inputs], columns=FEATURE_NAMES)
    X_scaled = scaler.transform(X)

    details = {}
    chart_labels = []
    chart_values = []

    for name, clf in models.items():
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_scaled)[0][1]
        else:
            prob = float(clf.predict(X_scaled)[0])

        details[name] = f"{prob * 100:.1f}%"
        chart_labels.append(name)
        chart_values.append(round(prob * 100, 1))

    percent = round(sum(chart_values) / len(chart_values), 1)
    acc_labels = list(accuracies.keys())
    acc_values = list(accuracies.values())

    return percent, details, chart_labels, chart_values, acc_labels, acc_values


def make_readable_inputs(inputs):
    readable_inputs = {}

    for k, v in inputs.items():
        label = DISPLAY_NAMES.get(k, k)

        if k in MAPPINGS:
            readable_inputs[label] = MAPPINGS[k].get(int(v), v)
        else:
            readable_inputs[label] = float(v) if k == "oldpeak" else int(v)

    return readable_inputs


def render_prediction_result(inputs, ecg_file=None, ecg_summary=None):
    percent, details, chart_labels, chart_values, acc_labels, acc_values = make_prediction(inputs)
    readable_inputs = make_readable_inputs(inputs)

    return render_template(
        "result.html",
        inputs=readable_inputs,
        percent=percent,
        details=details,
        chart_labels=chart_labels,
        chart_values=chart_values,
        acc_labels=acc_labels,
        acc_values=acc_values,
        ecg_file=ecg_file,
        ecg_summary=ecg_summary,
    )


def extract_from_uploaded_file(file, force_gemini_file=False):
    filename = file.filename.lower()
    mime_type = file.mimetype or "application/octet-stream"

    if force_gemini_file:
        file_bytes = file.read()

        if filename.endswith(".pdf"):
            mime_type = "application/pdf"
        elif filename.endswith(".png"):
            mime_type = "image/png"
        elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
            mime_type = "image/jpeg"

        return extract_features_from_report(file_bytes=file_bytes, mime_type=mime_type)

    if filename.endswith((".png", ".jpg", ".jpeg")):
        file_bytes = file.read()

        if filename.endswith(".png"):
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg"

        return extract_features_from_report(file_bytes=file_bytes, mime_type=mime_type)

    if filename.endswith(".pdf"):
        import pdfplumber

        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        if text.strip():
            return extract_features_from_report(report_text=text)

        file.seek(0)
        file_bytes = file.read()
        return extract_features_from_report(file_bytes=file_bytes, mime_type="application/pdf")

    report_text = file.read().decode("utf-8", errors="ignore")
    return extract_features_from_report(report_text=report_text)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    action = request.form.get("action")
    inputs = {}

    if action == "manual":
        try:
            for f in FEATURE_NAMES:
                val = request.form.get(f)

                if val is None or val == "":
                    return f"Missing value for {f}", 400

                inputs[f] = float(val) if f == "oldpeak" else int(float(val))

            return render_prediction_result(inputs)

        except Exception as e:
            return f"Invalid input: {e}", 400

    elif action == "file":
        file = request.files.get("report_file")

        if not file or file.filename == "":
            return "No file uploaded.", 400

        try:
            features = extract_from_uploaded_file(file, force_gemini_file=False)
            print("Extracted file features:", features)

            inputs = coerce_features(features)
            return render_prediction_result(inputs)

        except Exception as e:
            return f"Error: {e}", 400

    elif action == "ecg":
        file = request.files.get("report_file")

        if not file or file.filename == "":
            return "No ECG file uploaded.", 400

        filename = file.filename.lower()

        if not filename.endswith((".pdf", ".png", ".jpg", ".jpeg")):
            return "Invalid ECG file type. Upload PDF, PNG, JPG, or JPEG.", 400

        try:
            features = extract_from_uploaded_file(file, force_gemini_file=True)
            print("Extracted ECG features:", features)

            inputs = coerce_features(features)
            return render_prediction_result(
                inputs,
                ecg_file=file.filename,
                ecg_summary="ECG report processed successfully.",
            )

        except Exception as e:
            return f"ECG processing error: {e}", 400

    else:
        return "Unknown action.", 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
