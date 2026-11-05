
import os
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NoiseGuard AI",
    page_icon="⚙️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #f1f5f9;
}
h1, h2, h3 {
    color: #e2e8f0 !important;
}
p, label, span {
    color: #cbd5e1;
}
div[data-testid="stMetric"] {
    background: #172338;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #293b55;
}
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    baseline = joblib.load(os.path.join(BASE_DIR, "baseline_model.pkl"))
    robust = joblib.load(os.path.join(BASE_DIR, "robust_model.pkl"))
    return baseline, robust

@st.cache_data
def load_signal():
    return np.load(os.path.join(BASE_DIR, "demo_signal.npy"))

baseline_model, robust_model = load_models()
original_signal = load_signal().flatten()

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))

    features = [
        np.mean(signal),
        np.std(signal),
        rms,
        peak,
        kurtosis(signal),
        skew(signal),
        peak / (rms + 1e-10)
    ]

    return np.nan_to_num(
        np.array(features).reshape(1, -1)
    )

# ---------------- NOISE FUNCTIONS ----------------
def gaussian_noise(signal, snr_db=10):
    power = np.mean(signal ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    noise = np.random.normal(
        0, np.sqrt(noise_power), len(signal)
    )
    return signal + noise

def spike_noise(signal):
    noisy = signal.copy()
    count = max(1, int(0.02 * len(signal)))
    indices = np.random.choice(
        len(signal), count, replace=False
    )
    noisy[indices] += (
        np.random.choice([-1, 1], count)
        * 5 * np.std(signal)
    )
    return noisy

def drift_noise(signal):
    drift = np.linspace(
        0, 2 * np.std(signal), len(signal)
    )
    return signal + drift

def dropout_noise(signal):
    noisy = signal.copy()
    count = max(1, int(0.05 * len(signal)))
    start = np.random.randint(
        0, max(1, len(signal) - count + 1)
    )
    noisy[start:start + count] = 0
    return noisy

# ---------------- APPLY MULTIPLE NOISES ----------------
def apply_noise(signal, selected_noises, snr):
    noisy = signal.copy()

    for noise in selected_noises:
        if noise == "Gaussian":
            noisy = gaussian_noise(noisy, snr)

        elif noise == "Spike":
            noisy = spike_noise(noisy)

        elif noise == "Drift":
            noisy = drift_noise(noisy)

        elif noise == "Dropout":
            noisy = dropout_noise(noisy)

    return noisy

# ---------------- HEADER ----------------
st.title("⚙️ NoiseGuard AI")
st.subheader("Noise-Robust Machine Learning for Factory Sensor Data")

st.markdown(
    "An intelligent bearing fault detection system "
    "using vibration sensor data."
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Machine Learning Model",
    ["Baseline Model", "Noise-Robust Model"]
)

all_noises = ["Gaussian", "Spike", "Drift", "Dropout"]

st.sidebar.subheader("Noise Configuration")

apply_all = st.sidebar.checkbox(
    "Apply ALL noise types together"
)

selected_noises = st.sidebar.multiselect(
    "Select one or more noise types",
    options=all_noises,
    default=["Gaussian"]
)

if apply_all:
    selected_noises = all_noises

snr = st.sidebar.selectbox(
    "Gaussian Noise SNR",
    [20, 10, 5],
    index=1
)

st.sidebar.caption(
    "You can select multiple noise types or enable "
    "'Apply ALL noise types together'."
)

# ---------------- SIGNAL GENERATION ----------------
if selected_noises:
    noisy_signal = apply_noise(
        original_signal,
        selected_noises,
        snr
    )
else:
    noisy_signal = original_signal.copy()

# ---------------- SIGNAL DISPLAY ----------------
st.subheader("📡 Sensor Signal Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Original Vibration Signal**")
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(original_signal, linewidth=0.7)
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.2)
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.markdown("**Processed / Noisy Signal**")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(noisy_signal, linewidth=0.7)
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Amplitude")
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown(
    "**Applied noise:** "
    + (", ".join(selected_noises) if selected_noises else "None")
)

st.divider()

# ---------------- PREDICTION ----------------
st.subheader("🔍 Bearing Fault Prediction")

if st.button("🚀 Run Prediction", use_container_width=True):

    model = (
        baseline_model
        if model_choice == "Baseline Model"
        else robust_model
    )

    features = extract_features(noisy_signal)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    confidence = np.max(probabilities) * 100

    st.success("Prediction completed successfully!")

    result1, result2 = st.columns(2)

    with result1:
        st.metric("Predicted Condition", str(prediction))

    with result2:
        st.metric("Model Confidence", f"{confidence:.2f}%")

    st.subheader("📊 Class Probabilities")

    probability_data = {
        str(label): float(prob * 100)
        for label, prob in zip(model.classes_, probabilities)
    }

    st.bar_chart(probability_data)

    st.caption(
        "Probability scores are model outputs and "
        "should not be interpreted as calibrated certainty."
    )

st.divider()

# ---------------- FOOTER ----------------
st.info(
    "Demonstration using a saved CWRU vibration sample "
    "with simulated noise. This is not a live factory sensor."
)

st.caption("NoiseGuard AI | MCA Project")
