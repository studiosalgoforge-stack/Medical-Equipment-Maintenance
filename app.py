import flask
from flask import Flask, render_template_string, request, jsonify
import joblib
import numpy as np
import pandas as pd  # We need pandas to create the input for the preprocessor
import time
import os
import sys

app = Flask(__name__)

# --- Configuration ----

# This MUST match the model file you just trained
MODEL_PATH = "trained_breakdown_classifier.pkl"

# --- Load Model & Preprocessor (CRITICAL) ----
if not os.path.exists(MODEL_PATH):
    print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
    print("Please make sure you have run 'train_classifier.py' first.")
    sys.exit(1)

print(f"Loading model and preprocessor from {MODEL_PATH}...")
# The .pkl file is a dictionary containing BOTH the preprocessor and the model
pipeline_objects = joblib.load(MODEL_PATH)
preprocessor = pipeline_objects['preprocessor']
model = pipeline_objects['model']
print("Model loaded successfully.")

# ---- Helpers ----

def predict_breakdown(device_name, usage_hours, temperature, error_count):
    """
    Returns a full diagnostic report based on breakdown classification.
    """
    
    # --- 1. Create Input DataFrame (CRITICAL) ---
    # The preprocessor was trained on a DataFrame with specific column names.
    # We MUST create a new DataFrame with the same names for prediction.
    input_data = pd.DataFrame({
        'usage_hours': [float(usage_hours)],
        'temperature': [float(temperature)],
        'error_count': [int(error_count)],
        'device_name': [device_name] 
    })

    # --- 2. Preprocess the Data ---
    # Use the loaded preprocessor to transform the data
    # (e.g., one-hot encode 'device_name')
    X_processed = preprocessor.transform(input_data)

    # --- 3. Get Prediction & Probability ---
    # Predict the class (0 = Healthy, 1 = Breakdown)
    prediction_class = model.predict(X_processed)[0]
    
    # Predict the probability of each class [prob_of_0, prob_of_1]
    probabilities = model.predict_proba(X_processed)[0]
    
    # Get the probability of "Breakdown" (class 1)
    fail_probability = probabilities[1]
    fail_prob_percent = fail_probability * 100

    # --- 4. Create Diagnostic Report ---
    findings = []
    next_steps = []

    # A. Analyze Failure Probability (The new logic)
    if fail_probability > 0.5:
        status_label = "Critical"
        color = "#dc3545" # Red
        findings.append(f"CRITICAL RISK: {fail_prob_percent:.1f}% chance of imminent failure.")
        next_steps.append("Schedule immediate inspection and cease operation if possible.")
    elif fail_probability > 0.1:
        status_label = "High Risk"
        color = "#fd7e14" # Orange
        findings.append(f"HIGH RISK: {fail_prob_percent:.1f}% chance of failure detected.")
        next_steps.append("Schedule preventative maintenance within the next 1-2 weeks.")
    elif fail_probability > 0.02:
        status_label = "Elevated"
        color = "#ffc107" # Yellow
        findings.append(f"ELEVATED RISK: {fail_prob_percent:.1f}% chance of failure detected.")
        next_steps.append("Monitor machine closely. Check for physical anomalies.")
    else:
        status_label = "Low Risk"
        color = "#198754" # Green
        findings.append(f"LOW RISK: {fail_prob_percent:.1f}% chance of failure. Machine is healthy.")

    # B. Analyze Input Sensors (This logic is still great!)
    if float(temperature) > 90.0:
        findings.append("HIGH TEMP: Machine is running dangerously hot ({:.1f}°C).")
        next_steps.append("Check cooling systems regardless of failure risk.")
    elif float(temperature) > 60.0:
        findings.append("ELEVATED TEMP: Machine is running warm ({:.1f}°C).")
    
    if int(error_count) > 10:
        findings.append(f"HIGH ERROR COUNT: {error_count} errors logged.")
        next_steps.append("Run diagnostics and check error logs.")
    elif int(error_count) > 0:
        findings.append(f"{error_count} minor errors logged.")

    if float(usage_hours) > 8000:
        findings.append(f"HIGH USAGE: {usage_hours} hours. Nearing end-of-life for some components.")

    if not next_steps:
        next_steps.append("No immediate action required. Continue routine monitoring.")

    return {
        "prediction_class": int(prediction_class), # 0 or 1
        "probability_percent": f"{fail_prob_percent:.1f}%",
        "status_label": status_label,
        "color": color,
        "findings": findings,
        "next_steps": next_steps
    }

# ---- HTML Template ----

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Medical Machine Prediction Model</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root{
      --bg:#f6fbfc; --card:#ffffff; --accent:#0d6efd; --soft:#e9f7fb;
      --muted:#6c757d; --success:#198754; --danger:#dc3545; --warm:#fd7e14; --warning: #ffc107;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    body{ background:var(--bg); color:#0b2b36; margin:0; padding:24px;}
    .container{ max-width:1100px; margin:0 auto; }
    header{ display:flex; align-items:center; gap:16px; margin-bottom:18px;}
    .brand{ display:flex; flex-direction:column; }
    h1{ margin:0; font-size:20px; color: #063244; }
    p.lead{ margin:0; color:var(--muted); font-size:13px }
    .grid{ display:grid; grid-template-columns: 380px 1fr; gap:18px; align-items:start; }
    .card{ background:var(--card); border-radius:12px; padding:20px; box-shadow: 0 6px 18px rgba(6,18,22,0.04); }
    label{ font-size:13px; display:block; margin-bottom:6px; color:#0b2b36; font-weight: 500;}
    select, input { width:100%; padding:8px 10px; border-radius:8px; border:1px solid #e1eef2; background:transparent; box-sizing: border-box; }
    input[type=range] { padding: 0; }
    .small{ font-size:12px; color:var(--muted); margin-top:6px; }
    button{ border:0; padding:12px 16px; border-radius:10px; cursor:pointer; background:var(--accent); color:white; font-weight:600; width:100%;}
    .muted{ color:var(--muted); font-size:13px }
    .status { font-weight:700; padding:8px 12px; border-radius:10px; display:inline-block; color:white; }
    .row{ display:flex; gap:12px; }
    .col{ flex:1; }
    .form-group{ margin-top:14px; }
    footer{ margin-top:18px; color:var(--muted); font-size:13px; text-align:center; }
    /* New Report Styles */
    .report-card h3 { margin: 0 0 12px 0; }
    .report-section { margin-top: 16px; }
    .report-section h4 { margin: 0 0 8px 0; color: var(--accent); }
    .report-section ul { margin: 0; padding-left: 20px; }
    .report-section li { margin-bottom: 6px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div style="width:48px;height:48px;border-radius:12px;background:linear-gradient(135deg,var(--soft),#e0f7ff);display:flex;align-items:center;justify-content:center;">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M12 2v20" stroke="#0d6efd" stroke-width="1.6" stroke-linecap="round"/><path d="M5 7h14" stroke="#0d6efd" stroke-width="1.6" stroke-linecap="round"/></svg>
      </div>
      <div class="brand">
        <h1>Medical Equipment Failure Prediction</h1>
        <p class="lead">Interactive Diagnostic Tool</p>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        
        <div class="row">
          <div class="col form-group" style="margin-top:0;">
            <label for="machine_id">Machine ID</label>
            <input id="machine_id" type="text" value="ECG-MON-1138">
          </div>
          <div class="col form-group" style="margin-top:0;">
            <label for="location">Location</label>
            <input id="location" type="text" value="Ward 3, Bed 7">
          </div>
        </div>
        
        <div class="form-group">
          <label for="device">Device Type (Used by Model)</label>
          <select id="device">
            </select>
        </div>

        <div class="form-group">
          <label>Usage Hours</label>
          <input id="usage" type="range" min="0" max="10000" step="1" value="1200">
          <div class="row small" style="margin-top:6px;">
            <div class="col"><input id="usage_num" type="number" min="0" max="100000" step="1" value="1200"></div>
            <div class="col muted" style="text-align:right;">hours</div>
          </div>
        </div>

        <div class="form-group">
          <label>Temperature (°C)</label>
          <input id="temp" type="range" min="0" max="120" step="0.1" value="45.0">
          <div class="row small" style="margin-top:6px;">
            <div class="col"><input id="temp_num" type="number" min="0" max="150" step="0.1" value="45.0"></div>
            <div class="col muted" style="text-align:right;">°C</div>
          </div>
        </div>

        <div class="form-group">
          <label>Error Count (in last 24h)</label>
          <input id="errors" type="range" min="0" max="50" step="1" value="1">
          <div class="row small" style="margin-top:6px;">
            <div class="col"><input id="errors_num" type="number" min="0" max="100" step="1" value="1"></div>
            <div class="col muted" style="text-align:right;">errors</div>
          </div>
        </div>

        <div class="form-group" style="margin-top: 20px;">
          <button id="predictBtn">Run Diagnostic</button>
        </div>

        <div class="small muted" style="margin-top:12px;">
          Model file: <strong>{{model_name}}</strong>
        </div>
      </div>

      <div class="card report-card">
        <h3>Diagnostic Report</h3>
        <div class="muted small" style="margin-bottom: 16px;">
          For Machine: <strong id="report_machine_id">ECG-MON-1138</strong> at <strong id="report_location">Ward 3, Bed 7</strong><br>
          Report generated: <span id="lastUpdate">—</span>
        </div>

        <div style="display:flex;align-items:center;gap:12px; margin-bottom: 20px;">
          <div id="statusBadge" class="status" style="background:var(--muted);">—</div>
          <div>
            <div style="font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px;">Failure Risk</div>
            <div id="probText" style="font-weight:700; font-size:24px; color: #063244;">—</div>
            <div id="classText" class="muted small">—</div>
          </div>
        </div>

        <div id="report-content">
          <div class="report-section">
            <h4>Findings</h4>
            <ul id="report_findings">
              <li class="muted">Click 'Run Diagnostic' to generate a report.</li>
            </ul>
          </div>
          <div class="report-section">
            <h4>Next Steps</h4>
            <ul id="report_next_steps">
              <li class="muted">---</li>
            </ul>
          </div>
        </div>
        
      </div>
    </div>

    <footer>
      Model inference runs locally on the server. Make sure the model matches the training features.
    </footer>
  </div>

<script>
  // --- This data is from your preprocessor. Do not change it. ---
  const DEVICE_NAMES = {{ device_names | tojson }};

  // DOM elements
  const usage = document.getElementById('usage');
  const usage_num = document.getElementById('usage_num');
  const temp = document.getElementById('temp');
  const temp_num = document.getElementById('temp_num');
  const errors = document.getElementById('errors');
  const errors_num = document.getElementById('errors_num');

  // sync sliders and numbers
  function syncRange(range, num){
    range.addEventListener('input', ()=>{ num.value = range.value; });
    num.addEventListener('input', ()=>{ range.value = num.value; });
  }
  syncRange(usage, usage_num);
  syncRange(temp, temp_num);
  syncRange(errors, errors_num);

  const deviceSelect = document.getElementById('device');
  const predictBtn = document.getElementById('predictBtn');
  const statusBadge = document.getElementById('statusBadge');
  
  // New report elements
  const probText = document.getElementById('probText');
  const classText = document.getElementById('classText');
  const lastUpdate = document.getElementById('lastUpdate');
  const report_machine_id = document.getElementById('report_machine_id');
  const report_location = document.getElementById('report_location');
  const report_findings = document.getElementById('report_findings');
  const report_next_steps = document.getElementById('report_next_steps');

  // server predict
  async function getPrediction(payload){
    const resp = await fetch('/predict', {
      method:'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    return await resp.json();
  }

  predictBtn.addEventListener('click', async ()=>{
    const payload = {
      device: deviceSelect.value,
      usage_hours: usage.value,
      temperature: temp.value,
      error_count: errors.value
    };
    const res = await getPrediction(payload);
    handlePredictionResponse(res);
  });

  // --- NEW RESPONSE HANDLER ---
  function handlePredictionResponse(res){
    const now = new Date().toLocaleString();
    lastUpdate.textContent = now;

    // Update report header
    report_machine_id.textContent = document.getElementById('machine_id').value;
    report_location.textContent = document.getElementById('location').value;

    // Update prediction box
    statusBadge.textContent = res.status_label;
    statusBadge.style.background = res.color;
    probText.textContent = res.probability_percent;
    classText.textContent = `(Predicted Class: ${res.prediction_class})`;

    // Update report content
    report_findings.innerHTML = res.findings.map(f => `<li>${f}</li>`).join('');
    report_next_steps.innerHTML = res.next_steps.map(s => `<li>${s}</li>`).join('');
  }
  
  // --- Populate Device Dropdown ---
  function populateDevices(){
    for (const device of DEVICE_NAMES) {
      const option = document.createElement('option');
      option.value = device;
      option.textContent = device;
      deviceSelect.appendChild(option);
    }
  }

  // initial prediction on load
  window.addEventListener('load', ()=>{
    populateDevices();
    predictBtn.click();
  });
</script>
</body>
</html>
"""

# ---- Flask Routes ----

@app.route("/")
def index():
    # Get device names from the preprocessor to populate the dropdown
    try:
        device_names = list(preprocessor.named_transformers_['cat'].categories_[0])
    except Exception as e:
        print(f"Warning: Could not get device names from preprocessor. Using defaults. Error: {e}")
        device_names = ["ECG Monitor", "Ventilator", "Infusion Pump"] # Fallback
        
    return render_template_string(
        TEMPLATE, 
        model_name=os.path.basename(MODEL_PATH),
        device_names=device_names
    )

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json(force=True)
        
        result = predict_breakdown(
            device_name=data.get("device", "ECG Monitor"),
            usage_hours=data.get("usage_hours", 1000),
            temperature=data.get("temperature", 40.0),
            error_count=data.get("error_count", 0)
        )
        
        result['time'] = time.strftime("%H:%M:%S")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Get port from environment variable or default to 5001
    port = int(os.environ.get("PORT", 5001))
    # Run the app
    app.run(host="0.0.0.0", port=port, debug=True)