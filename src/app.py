import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EEG Brain Demo")

st.title("🧠 EEG Brain State Detector")

# -------------------------
# 1. EEG GENERATION PROPRE
# -------------------------
def generate_eeg():
    mode = np.random.choice(["REST", "ACTIVE"], p=[0.5, 0.5])

    if mode == "REST":
        eeg = np.random.randn(32, 128) * 0.25  # signal calme
    else:
        eeg = np.random.randn(32, 128) * 1.0 + np.sin(np.linspace(0, 5, 128))

    return eeg

# -------------------------
# 2. DETECTION LOGIQUE
# -------------------------
def detect_state(eeg):
    activity = np.std(eeg)

    if activity < 0.5:
        return "REST 💤", activity
    else:
        return "ACTIVE ⚡", activity

# -------------------------
# 3. UI
# -------------------------
if st.button("🔄 ANALYZE BRAIN SIGNAL"):

    eeg = generate_eeg()

    state, activity = detect_state(eeg)

    # ---------------- EEG CURVE ----------------
    fig, ax = plt.subplots()
    ax.plot(eeg[0])
    ax.set_title("EEG Signal (Channel 1)")
    ax.set_ylim(-3, 3)

    st.pyplot(fig)

    # ---------------- RESULT ----------------
    st.write("📊 Activity level:", round(activity, 3))

    if "REST" in state:
        st.success(f"🧠 State: {state}")
    else:
        st.error(f"🧠 State: {state}")