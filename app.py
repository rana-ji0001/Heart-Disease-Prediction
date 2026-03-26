# app.py

from flask import Flask, render_template, request
from dotenv import load_dotenv
import joblib
import pandas as pd
import numpy as np
import os
import json

load_dotenv()

app = Flask(__name__)


from google import genai
from google.genai.types import HttpOptions

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"), http_options=HttpOptions(api_version="v1beta")
)
for model in client.models.list():
    print(model.name)


# Load saved artifact
artifact = joblib.load("model.pkl")
models = artifact["models"]
scaler = artifact["scaler"]
FEATURE_NAMES = artifact["features"]
accuracies = artifact["accuracies"]


import json


def extract_features_from_report(report_text):
    """
    Uses Gemini to extract required features for heart disease prediction.
    """
    prompt = f"""
    You are a medical data assistant that converts a patient’s health report into features for a machine learning model.
    From the report below, extract and output ONLY a JSON object with these exact keys:
    {FEATURE_NAMES}

    Return ONLY valid JSON — no explanations, no text, no markdown.
    Patient report:
    {report_text}
    """

    try:
        # UPDATED: Use a model from your working list.
        # 'gemini-2.0-flash' is highly recommended for this.
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )

        # In the new SDK, use .text to get the content
        result_text = response.text.strip()

    except Exception as e:
        # Remove the nested "gemini-1.0-pro" call entirely as it will always fail.
        raise ValueError(f"Gemini API Error: {e}")

    # --- Robust JSON Cleaning ---
    # Removes markdown code blocks if the AI accidentally includes them
    if "```" in result_text:
        result_text = result_text.split("```")[1]
        if result_text.lower().startswith("json"):
            result_text = result_text[4:].strip()

    try:
        features = json.loads(result_text)
    except json.JSONDecodeError:
        # Fallback if the AI output is messy
        import re

        json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if json_match:
            features = json.loads(json_match.group())
        else:
            raise ValueError(f"Failed to parse JSON: {result_text}")

    if isinstance(features, list):
        features = features[0]

    # Ensure data types match what your 'model.pkl' expects
    final_features = {}
    for f in FEATURE_NAMES:
        val = features.get(f, 0)
        try:
            if f == "oldpeak":
                final_features[f] = float(val)
            else:
                final_features[f] = int(float(val))
        except (ValueError, TypeError):
            final_features[f] = 0  # Default to 0 if extraction fails

    return final_features


# Mappings for categorical fields
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

# Friendly display names
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
    action = request.form.get(
        "action"
    )  # Detect which button was pressed: 'manual' or 'file'
    inputs = {}

    # -----------------------------
    # 1️⃣ FILE UPLOAD HANDLING
    # -----------------------------

    if action == "file":
        file = request.files.get("report_file")
        if not file or file.filename == "":
            return "No file uploaded.", 400

        try:
            # 1) Read uploaded file
            report_text = file.read().decode("utf-8")

            # 2) Extract features using GPT
            features = extract_features_from_report(report_text)
            print("🔍 Extracted features from report:", features)

            # 2b) If GPT returned a list of records, take the first
            if isinstance(features, list):
                print(
                    f"⚠️ GPT returned list of {len(features)} records — using the first one."
                )
                features = features[0]

            # 3) Coerce feature values to numeric types expected by the model
            coerced = {}
            for f in FEATURE_NAMES:
                raw = features.get(f, None)
                if raw is None:
                    # missing -> default to 0
                    coerced[f] = 0.0 if f == "oldpeak" else 0
                    continue
                # oldpeak may be float
                if f == "oldpeak":
                    try:
                        coerced[f] = float(raw)
                    except Exception:
                        coerced[f] = 0.0
                else:
                    # other numeric fields -> int
                    try:
                        coerced[f] = int(float(raw))
                    except Exception:
                        # fallback: if it's already numeric
                        try:
                            coerced[f] = int(raw)
                        except Exception:
                            coerced[f] = 0

            # 4) Prepare dataframe in the right column order
            X = pd.DataFrame([coerced], columns=FEATURE_NAMES)

            # 5) Scale
            X_scaled = scaler.transform(X)

            # 6) Predict with each model and build details
            details = {}
            chart_labels = []
            chart_values = []
            for name, clf in models.items():
                # some models may not have predict_proba; handle that gracefully
                if hasattr(clf, "predict_proba"):
                    prob = clf.predict_proba(X_scaled)[0][1]
                else:
                    # fallback to predict (0/1) — not ideal for probability but prevents crash
                    pred = clf.predict(X_scaled)[0]
                    prob = float(pred)
                details[name] = f"{prob*100:.1f}%"
                chart_labels.append(name)
                chart_values.append(round(prob * 100, 1))

            # 7) Average probability and accuracy data
            avg_prob = (sum(chart_values) / len(chart_values)) if chart_values else 0.0
            percent = round(avg_prob, 1)

            acc_labels = list(accuracies.keys())
            acc_values = list(accuracies.values())

            # 8) Convert to readable inputs using your existing MAPPINGS and DISPLAY_NAMES
            readable_inputs = {}
            for k, v in coerced.items():
                label = DISPLAY_NAMES.get(k, k)
                if k in MAPPINGS:
                    # MAPPINGS uses numeric keys like {0:"Normal", 1:"Abnormal"}
                    # we must look up using numeric v
                    readable_inputs[label] = MAPPINGS[k].get(v, v)
                else:
                    # For oldpeak show decimal
                    if k == "oldpeak":
                        readable_inputs[label] = float(v)
                    else:
                        readable_inputs[label] = int(v)

            # 9) Render the same template as manual route
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
            # Return the error and also print stack for debugging
            import traceback

            traceback.print_exc()
            return f"Error processing file: {e}", 400
    # -----------------------------
    # 2️⃣ MANUAL INPUT HANDLING
    # -----------------------------
    elif action == "manual":
        try:
            for f in FEATURE_NAMES:
                val = request.form.get(f)
                if val is None or val == "":
                    return f"Missing value for {f}", 400
                if f == "oldpeak":
                    inputs[f] = float(val)
                else:
                    inputs[f] = int(float(val))

            X = pd.DataFrame([inputs], columns=FEATURE_NAMES)
            X_scaled = scaler.transform(X)

        except Exception as e:
            return f"Invalid input: {e}", 400

    else:
        return "Unknown action.", 400

    # -----------------------------
    # 3️⃣ MODEL PREDICTION
    # -----------------------------
    details = {}
    chart_labels = []
    chart_values = []

    for name, clf in models.items():
        prob = clf.predict_proba(X_scaled)[0][1]
        details[name] = f"{prob*100:.1f}%"
        chart_labels.append(name)
        chart_values.append(round(prob * 100, 1))

    avg_prob = sum(chart_values) / len(chart_values)
    percent = round(avg_prob, 1)

    acc_labels = list(accuracies.keys())
    acc_values = list(accuracies.values())

    # -----------------------------
    # 4️⃣ CONVERT TO READABLE INPUTS
    # -----------------------------
    readable_inputs = {}

    # Use same logic for both manual and file input
    for k, v in inputs.items():
        label = DISPLAY_NAMES.get(k, k)
        if k in MAPPINGS:
            readable_inputs[label] = MAPPINGS[k].get(v, v)
        else:
            readable_inputs[label] = v
    # readable_inputs = {"Source": "Uploaded Report File (AI-extracted)"}

    # -----------------------------
    # 5️⃣ RENDER RESULT
    # -----------------------------
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
