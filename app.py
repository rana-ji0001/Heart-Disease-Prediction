# app.py

from flask import Flask, render_template, request
from dotenv import load_dotenv
import joblib
import pandas as pd
import numpy as np
import os
import json
import base64
import time

load_dotenv()

app = Flask(__name__)

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load saved artifact
artifact = joblib.load("model.pkl")
models = artifact["models"]
scaler = artifact["scaler"]
FEATURE_NAMES = artifact["features"]
accuracies = artifact["accuracies"]


# 🔁 RETRY FUNCTION (fix 503 errors)
def call_gemini_with_retry(make_request, retries=3):
    for i in range(retries):
        try:
            return make_request()
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"⚠️ Retry {i+1} due to high demand...")
                time.sleep(3 + i * 2)  # increasing delay
            else:
                raise e
    raise ValueError("Gemini failed after retries")


def extract_features_from_report(report_text=None, image_file=None):
    """
    Extract features using Gemini (text + ECG image)
    """

    # 🔒 LIMIT INPUT SIZE (TPM protection)
    MAX_CHARS = 20000
    if report_text and len(report_text) > MAX_CHARS:
        report_text = report_text[:MAX_CHARS]

    prompt = f"""
Extract medical features from the report.

Return ONLY a valid JSON object with these keys:
{FEATURE_NAMES}

Rules:
- No explanation
- No markdown
- Only JSON
- Use numbers only
"""

    try:
        # -------------------------
        # 🧠 IMAGE (ECG)
        # -------------------------
        if image_file:
            image_bytes = image_file.read()

            def make_request():
                # 🔄 fallback models
                for m in ["gemini-2.5-flash", "gemini-1.5-flash"]:
                    try:
                        return client.models.generate_content(
                            model=m,
                            contents=[
                                prompt,
                                {
                                    "mime_type": "image/png",
                                    "data": image_bytes
                                }
                            ]
                        )
                    except Exception as e:
                        print(f"{m} failed, trying next...")
                raise ValueError("All Gemini models failed")

            response = call_gemini_with_retry(make_request)

        # -------------------------
        # 🧠 TEXT
        # -------------------------
        else:
            prompt += f"\nPatient report:\n{report_text}"

            def make_request():
                for m in ["gemini-2.5-flash", "gemini-1.5-flash"]:
                    try:
                        return client.models.generate_content(
                            model=m,
                            contents=prompt
                        )
                    except Exception as e:
                        print(f"{m} failed, trying next...")
                raise ValueError("All Gemini models failed")

            response = call_gemini_with_retry(make_request)

        result_text = response.text.strip()

        # 🧹 Clean JSON
        if "```" in result_text:
            result_text = result_text.split("```")[1]
            if result_text.lower().startswith("json"):
                result_text = result_text[4:].strip()

        # 🛑 RATE LIMIT SAFE (free tier)
        time.sleep(8)

    except Exception as e:
        raise ValueError(f"Gemini API Error: {e}")

    try:
        features = json.loads(result_text)
    except:
        import re
        match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if match:
            features = json.loads(match.group())
        else:
            raise ValueError(f"JSON parse failed: {result_text}")

    if isinstance(features, list):
        features = features[0]

    # Type safety
    final_features = {}
    for f in FEATURE_NAMES:
        val = features.get(f, 0)
        try:
            if f == "oldpeak":
                final_features[f] = float(val)
            else:
                final_features[f] = int(float(val))
        except:
            final_features[f] = 0

    return final_features


# -----------------------------
# MAPPINGS
# -----------------------------
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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    action = request.form.get("action")
    inputs = {}

    if action == "file":
        file = request.files.get("report_file")

        if not file or file.filename == "":
            return "No file uploaded.", 400

        try:
            filename = file.filename.lower()

            if filename.endswith((".png", ".jpg", ".jpeg")):
                # Image (ECG)
                features = extract_features_from_report(image_file=file)

            elif filename.endswith(".pdf"):
                # ✅ Handle PDF properly
                import pdfplumber

                with pdfplumber.open(file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""

                features = extract_features_from_report(report_text=text)

            else:
                # Text file
                report_text = file.read().decode("utf-8", errors="ignore")
                features = extract_features_from_report(report_text=report_text)

            print("🔍 Extracted features:", features)

            coerced = {}
            for f in FEATURE_NAMES:
                raw = features.get(f, None)

                if raw is None:
                    coerced[f] = 0.0 if f == "oldpeak" else 0
                    continue

                if f == "oldpeak":
                    try:
                        coerced[f] = float(raw)
                    except:
                        coerced[f] = 0.0
                else:
                    try:
                        coerced[f] = int(float(raw))
                    except:
                        coerced[f] = 0

            X = pd.DataFrame([coerced], columns=FEATURE_NAMES)
            X_scaled = scaler.transform(X)

            details = {}
            chart_labels = []
            chart_values = []

            for name, clf in models.items():
                if hasattr(clf, "predict_proba"):
                    prob = clf.predict_proba(X_scaled)[0][1]
                else:
                    prob = float(clf.predict(X_scaled)[0])

                details[name] = f"{prob*100:.1f}%"
                chart_labels.append(name)
                chart_values.append(round(prob * 100, 1))

            percent = round(sum(chart_values) / len(chart_values), 1)

            acc_labels = list(accuracies.keys())
            acc_values = list(accuracies.values())

            readable_inputs = {}
            for k, v in coerced.items():
                label = DISPLAY_NAMES.get(k, k)
                if k in MAPPINGS:
                    readable_inputs[label] = MAPPINGS[k].get(v, v)
                else:
                    readable_inputs[label] = float(v) if k == "oldpeak" else int(v)

            return render_template(
                "result.html",
                inputs=readable_inputs,
                percent=percent,
                details=details,
                chart_labels=chart_labels,
                chart_values=chart_values,
                acc_labels=acc_labels,
                acc_values=acc_values,
            )

        except Exception as e:
            return f"Error: {e}", 400

    elif action == "manual":
        try:
            for f in FEATURE_NAMES:
                val = request.form.get(f)
                if val is None or val == "":
                    return f"Missing value for {f}", 400

                inputs[f] = float(val) if f == "oldpeak" else int(float(val))

            X = pd.DataFrame([inputs], columns=FEATURE_NAMES)
            X_scaled = scaler.transform(X)

        except Exception as e:
            return f"Invalid input: {e}", 400

    else:
        return "Unknown action.", 400

    details = {}
    chart_labels = []
    chart_values = []

    for name, clf in models.items():
        prob = clf.predict_proba(X_scaled)[0][1]
        details[name] = f"{prob*100:.1f}%"
        chart_labels.append(name)
        chart_values.append(round(prob * 100, 1))

    percent = round(sum(chart_values) / len(chart_values), 1)

    acc_labels = list(accuracies.keys())
    acc_values = list(accuracies.values())

    readable_inputs = {}
    for k, v in inputs.items():
        label = DISPLAY_NAMES.get(k, k)
        readable_inputs[label] = MAPPINGS[k].get(v, v) if k in MAPPINGS else v

    return render_template(
        "result.html",
        inputs=readable_inputs,
        percent=percent,
        details=details,
        chart_labels=chart_labels,
        chart_values=chart_values,
        acc_labels=acc_labels,
        acc_values=acc_values,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)