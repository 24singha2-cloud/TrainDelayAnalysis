import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Train Delay Prediction", page_icon="🚆", layout="wide")
MODEL_PATH = Path(__file__).parent / "best_train_delay_model.pkl"

st.title("🚆 Train Delay Analysis and Prediction")
st.write("Enter journey and operating details to predict whether a journey is likely to be delayed.")
st.info("This app uses the trained project model and dataset. It does not use live train-running or weather data.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load model file: {MODEL_PATH}. Ensure best_train_delay_model.pkl is in the same folder as app.py. Details: {exc}")
    st.stop()

zones = ["NR", "WR", "ER", "SR", "CR", "SCR", "SWR", "NER", "NCR", "SER", "ECR", "WCR", "NFR", "SECR", "ECoR", "MR"]
train_types = ["Passenger", "Mail/Express", "Superfast", "Rajdhani", "Shatabdi", "Vande Bharat", "Duronto"]

with st.form("prediction_form"):
    st.subheader("Journey details")
    c1, c2, c3 = st.columns(3)
    with c1:
        zone = st.selectbox("Railway zone", zones, index=0)
        train_type = st.selectbox("Train type", train_types, index=1)
        month = st.selectbox("Month", list(range(1, 13)), index=6)
        day_of_week = st.selectbox("Day of week (0=Monday, 6=Sunday)", list(range(7)), index=2)
        departure_hour = st.slider("Departure hour (0–23)", 0, 23, 8)
    with c2:
        distance_km = st.number_input("Distance (km)", min_value=1.0, max_value=5000.0, value=500.0, step=10.0)
        scheduled_stops = st.number_input("Scheduled stops", min_value=0, max_value=100, value=5, step=1)
        congestion = st.slider("Zone congestion index", 0.0, 1.0, 0.5, 0.01)
        weather_risk = st.slider("Weather risk score", 0.0, 1.0, 0.3, 0.01)
        historical_ontime = st.slider("Historical on-time percentage", 0.0, 100.0, 75.0, 1.0)
    with c3:
        late_incoming = st.selectbox("Incoming rake late?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        is_electrified = st.selectbox("Electrified route?", [1, 0], format_func=lambda x: "Yes" if x else "No")
        track_doubled = st.selectbox("Double-tracked route?", [1, 0], format_func=lambda x: "Yes" if x else "No")
        loco_age = st.slider("Locomotive age (years)", 1, 30, 10)
        coach_age = st.slider("Coach age (years)", 1, 35, 8)
    submitted = st.form_submit_button("Predict train delay", type="primary")

if submitted:
    season = ("Winter" if month in [12, 1, 2] else "Summer" if month in [3, 4, 5] else "Monsoon" if month in [6, 7, 8, 9] else "Post-Monsoon")
    is_weekend = int(day_of_week in [5, 6])
    is_peak_hour = int(departure_hour in [7, 8, 9, 10, 17, 18, 19, 20])
    is_fog_risk = int(month in [12, 1, 2] and zone in ["NR", "NER", "NCR", "NFR"])
    is_monsoon_season = int(month in [6, 7, 8, 9])
    row = pd.DataFrame([{
        "train_number": 12345, "zone": zone, "train_type": train_type,
        "month": int(month), "season": season, "day_of_week": int(day_of_week),
        "is_weekend": is_weekend, "departure_hour": int(departure_hour), "is_peak_hour": is_peak_hour,
        "distance_km": float(distance_km), "scheduled_stops": int(scheduled_stops),
        "zone_congestion_index": float(congestion), "weather_risk_score": float(weather_risk),
        "is_fog_risk": is_fog_risk, "is_monsoon_season": is_monsoon_season,
        "late_incoming_rake": int(late_incoming), "historical_ontime_pct": float(historical_ontime),
        "is_electrified": int(is_electrified), "track_doubled": int(track_doubled),
        "loco_age_years": int(loco_age), "coach_age_years": int(coach_age)
    }])
    try:
        prediction = int(model.predict(row)[0])
        label = "Delayed" if prediction == 1 else "On Time"
        st.subheader("Prediction result")
        if prediction == 1:
            st.error(f"Predicted status: {label}")
        else:
            st.success(f"Predicted status: {label}")
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(row)[0][1])
            st.metric("Estimated model probability of delay", f"{probability:.1%}")
            st.caption("This is the model's estimated probability, not a guarantee or a live railway status.")
    except Exception as exc:
        st.error(f"Prediction failed. Check that the input columns match the model's training features. Details: {exc}")
