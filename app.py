
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

from scipy.stats import kurtosis, skew

st.set_page_config(
    page_title="NoiseGuard AI",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ NoiseGuard AI")
st.subheader("Noise-Robust Machine Learning for Factory Sensor Data")

st.write(
    "An intelligent bearing fault detection system "
    "using vibration sensor data."
)

# Load saved models and sample
baseline_model = joblib.load("baseline_model.pkl")
robust_model = joblib.load("robust_model.pkl")
original = np.load("demo_signal.npy")

# Noise functions
def gaussian_noise(signal, snr_db=10):
    power = np.mean(signal ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    noise = np.random.normal(
        0, np.sqrt(noise_power), len(signal)
    )
    return signal + noise

def spike_noise(signal):
    noisy = signal.copy()
    count = max(1, int(len(signal) * 0.02))
    indices = np.random.choice(len(signal), count, replace=False)
    noisy[indices] += np.random.choice(
        [-1, 1], count
    ) * 5 * np.std(signal)
    return noisy

def drift_noise(signal):
    return signal + np.linspace(
        0, 2 * np.std(signal), len(signal)
    )

def dropout_noise(signal):
    noisy = signal.copy()
    count = max(1, int(len(signal) * 0.05))
    start = np.random.randint(0, len(signal) - count + 1)
    noisy[start:start + count] = 0
    return noisy

def extract_features(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))

    return np.nan_to_num([[
        np.mean(signal),
        np.std(signal),
        rms,
        peak,
        kurtosis(signal),
        skew(signal),
        peak / (rms + 1e-10)
    ]])

# Sidebar settings
st.sidebar.header("Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Baseline Model", "Noise-Robust Model"]
)

noise_choice = st.sidebar.selectbox(
    "Select Noise Type",
    ["Clean", "Gaussian", "Spike", "Drift", "Dropout"]
)

snr = st.sidebar.selectbox(
    "Gaussian Noise SNR",
    [20, 10, 5],
    index=1
)

if model_choice == "Baseline Model":
    model = baseline_model
else:
    model = robust_model

# Apply noise
if noise_choice == "Gaussian":
    signal = gaussian_noise(original, snr)
elif noise_choice == "Spike":
    signal = spike_noise(original)
elif noise_choice == "Drift":
    signal = drift_noise(original)
elif noise_choice == "Dropout":
    signal = dropout_noise(original)
else:
    signal = original.copy()

st.write("### Sensor Signal")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(signal)
ax.set_xlabel("Samples")
ax.set_ylabel("Vibration Amplitude")
ax.set_title(noise_choice + " Sensor Signal")
st.pyplot(fig)

# Prediction
if st.button("Analyze Sensor Data"):
    features = extract_features(signal)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    confidence = np.max(probabilities) * 100

    st.write("### Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Condition", prediction)

    with col2:
        st.metric("Model Confidence", f"{confidence:.2f}%")

    st.write("### Class Probabilities")

    probability_data = {
        str(c): float(p)
        for c, p in zip(classes, probabilities)
    }

    st.bar_chart(probability_data)

    st.info(
        "This is a demonstration using a saved CWRU vibration "
        "sample with simulated noise, not live factory sensor data."
    )
