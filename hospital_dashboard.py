import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("trained_rul_model.pkl")

st.set_page_config(page_title="Hospital IoT & Predictive Maintenance Dashboard", layout="wide")
st.title("🏥 Real-Time Patient Monitoring + Medical Equipment Health Prediction")

# ---------------- LEFT SECTION: PATIENT VITALS ---------------- #
st.subheader("🩺 Patient Live Vitals Monitor")

patient_placeholder = st.empty()
heart_rate = 72
oxygen = 98
temp = 36.8

# ---------------- RIGHT SECTION: MACHINE PREDICTIVE MAINTENANCE ---------------- #
st.subheader("⚙️ Medical Equipment Health Status")

device = st.selectbox("Select Medical Device", [
    "Ventilator", "ECG Monitor", "Infusion Pump", "Ultrasound Scanner",
    "X-Ray Machine", "Defibrillator", "Patient Monitor", "Anesthesia Machine"
])

sim_interval = st.sidebar.number_input("Refresh Every (seconds)", min_value=1, max_value=10, value=2, step=1)

machine_placeholder = st.empty()

def generate_machine_data():
    return {
        "usage_hours": np.random.uniform(100, 9000),
        "temperature": np.random.uniform(25, 90),
        "error_count": np.random.randint(0, 10)
    }

history = []

while True:
    # ------------ PATIENT DATA UPDATE ------------ #
    heart_rate += np.random.randint(-2, 3)
    oxygen += np.random.randint(-1, 2)
    temp += round(np.random.uniform(-0.1, 0.1), 2)

    patient_data = pd.DataFrame([{
        "Heart Rate (BPM)": heart_rate,
        "Blood Oxygen (%)": oxygen,
        "Body Temperature (°C)": temp,
        "Time": time.strftime("%H:%M:%S")
    }])

    with patient_placeholder.container():
        st.dataframe(patient_data)

    # ------------ MACHINE DATA UPDATE ------------ #
    machine_data = generate_machine_data()
    df_machine = pd.DataFrame([machine_data])

    predicted_rul_days = model.predict(df_machine)[0]
    predicted_rul_years = round(predicted_rul_days / 365, 2)

    if predicted_rul_years > 5:
        status = "✅ Healthy - No Action Needed"
        color = "green"
    elif predicted_rul_years > 2:
        status = "⚠️ Warning - Schedule Maintenance"
        color = "orange"
    else:
        status = "🔴 CRITICAL - Failure Imminent!"
        color = "red"

    history.append([datetime.now().strftime("%H:%M:%S"), predicted_rul_years])

    with machine_placeholder.container():
        col1, col2 = st.columns(2)
        col1.metric("Remaining Useful Life", f"{predicted_rul_years} Years")
        col2.markdown(f"<h3 style='color:{color};'>{status}</h3>", unsafe_allow_html=True)

        st.write("📦 Live Machine Sensor Data:")
        st.dataframe(df_machine)

    time.sleep(sim_interval)
