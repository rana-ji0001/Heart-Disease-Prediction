# app.py

from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import json
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Load saved artifact
artifact = joblib.load("model.pkl")
models = artifact["models"]
scaler = artifact["scaler"]
FEATURE_NAMES = artifact["features"]
accuracies = artifact["accuracies"]

import openai

def extract_features_from_report(report_text):
    """
    Uses OpenAI GPT to extract required features for heart disease prediction.
    Returns a dictionary with FEATURE_NAMES as keys.
    """
    prompt = f"""
    Extract the following features from the patient report: 
    {FEATURE_NAMES}
    Return them in JSON format with keys exactly as above.
    Example output:
    {{"age": 52, "sex": 1, "cp": 0, ...}}

    Patient report:
    {report_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-4o / gpt-4-turbo if available
        messages=[
            {"role": "system", "content": "You are a medical data assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    result_text = response.choices[0].message.content.strip()

    try:
        features = json.loads(result_text)
    except Exception:
        raise ValueError("Failed to parse model output as JSON. Got:\n" + result_text)

    return features


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
    }
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
    "thal": "Thalassemia"
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    action = request.form.get("action")  # Detect which button was pressed: 'manual' or 'file'
    inputs = {}

    # -----------------------------
    # 1️⃣ FILE UPLOAD HANDLING
    # -----------------------------
    
    if action == "file":
        file = request.files.get("report_file")
        if not file or file.filename == "":
            return "No file uploaded.", 400

        try:
            # Read the uploaded file as text
            report_text = file.read().decode("utf-8")

            # Extract features using OpenAI GPT
            inputs = extract_features_from_report(report_text)

            # Convert to DataFrame
            X = pd.DataFrame([inputs], columns=FEATURE_NAMES)
            X_scaled = scaler.transform(X)

        except Exception as e:
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
    if action == "manual":
        for k, v in inputs.items():
            label = DISPLAY_NAMES.get(k, k)
            if k in MAPPINGS:
                readable_inputs[label] = MAPPINGS[k].get(v, v)
            else:
                readable_inputs[label] = v
    else:
        readable_inputs = {"Source": "Uploaded Report File (AI-extracted)"}

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
        acc_values=acc_values
    )


if __name__ == "__main__":
    app.run(debug=True)
