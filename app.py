# =============================================================================
#  FRAUD DETECTION - STREAMLIT DEPLOYMENT APP (FIXED VERSION)
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔒",
    layout="centered"
)

st.title("Credit Card Fraud Detection System")
st.write("Enter transaction details below to check if it is Legitimate or Fraudulent.")

# ── Load model + preprocessing objects ───────────────────────────────────────
@st.cache_resource
def load_objects():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("selector.pkl", "rb") as f:
        selector = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("selected_features.pkl", "rb") as f:
        selected_features = pickle.load(f)

    return model, selector, scaler, selected_features


try:
    model, selector, scaler, selected_features = load_objects()
    st.success("Model loaded successfully.")
except:
    st.error("Model files missing. Run training script first.")
    st.stop()


# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", 0.0, 100000.0, 100.0)
    time   = st.number_input("Time (seconds)", 0.0, 200000.0, 50000.0)

with col2:
    st.write("Principal Components (V1 - V10)")
    v_vals = {}
    for i in range(1, 11):
        v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v{i}")

st.write("Additional PCA Features (V11 - V28)")
cols = st.columns(3)

for i in range(11, 29):
    with cols[(i - 11) % 3]:
        v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v{i}")


# ── Threshold slider (PRO FEATURE) ────────────────────────────────────────────
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.5)


# ── Prediction ───────────────────────────────────────────────────────────────
if st.button("Predict", type="primary"):

    try:
        # Build raw input
        raw = {f"V{i}": v_vals[f"V{i}"] for i in range(1, 29)}

        # Scale Amount & Time properly
        scaled = scaler.transform([[amount, time]])
        raw["Amount_scaled"] = scaled[0][0]
        raw["Time_scaled"]   = scaled[0][1]

        # Feature Engineering (same as training)
        raw["V14_V17"] = raw["V14"] * raw["V17"]
        raw["V12_V10"] = raw["V12"] * raw["V10"]

        input_df = pd.DataFrame([raw])

        # Ensure correct feature order
        input_df = input_df[selected_features]

        # Apply feature selection
        input_sel = selector.transform(input_df)

        # Prediction
        prob = model.predict_proba(input_sel)[0][1]
        pred = int(prob > threshold)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # ── Results ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"🚨 FRAUDULENT TRANSACTION")
        st.write("⚠️ High risk transaction. Consider blocking or reviewing.")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION")
        st.write("✔️ This transaction appears safe.")

    st.metric("Fraud Probability", f"{prob * 100:.2f}%")

    # Risk label
    if prob < 0.3:
        risk = "LOW RISK"
    elif prob < 0.7:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK"

    st.info(f"Risk Level: {risk}")

    # Progress bar
    st.progress(float(prob))
    st.caption("0% = Safe  |  100% = Fraud")


# ── Sidebar info ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("""
    This app uses a **Tuned Random Forest model** trained on
    Credit Card Fraud dataset.

    ✔ Handles class imbalance (SMOTE)  
    ✔ Feature engineering applied  
    ✔ Feature selection (SelectKBest)  
    ✔ High AUC-ROC performance  

    Built for ML InnovateX Hackathon 🚀
    """)






















# # =============================================================================
# #  FRAUD DETECTION - STREAMLIT DEPLOYMENT APP
# # =============================================================================
# #
# #  Run this AFTER running fraud_detection.py (which saves the model files)
# #
# #  Install Streamlit: pip install streamlit
# #  Run app         : streamlit run app.py
# #
# #  This opens a web app at http://localhost:8501
# #  For deployment: https://share.streamlit.io (free, just push to GitHub)
# #
# # =============================================================================

# import streamlit as st
# import numpy as np
# import pickle
# import os

# # ── Page config ──────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Credit Card Fraud Detector",
#     page_icon="🔒",
#     layout="centered"
# )

# st.title("Credit Card Fraud Detection System")
# st.write("Enter transaction details below to check if it is Legitimate or Fraudulent.")

# # ── Load model ────────────────────────────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     with open("best_model.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("selector.pkl", "rb") as f:
#         selector = pickle.load(f)
#     return model, selector

# try:
#     model, selector = load_model()
#     st.success("Model loaded successfully.")
# except FileNotFoundError:
#     st.error("Model files not found. Please run fraud_detection.py first.")
#     st.stop()

# # ── Input form ────────────────────────────────────────────────────────────────
# st.subheader("Transaction Details")

# col1, col2 = st.columns(2)

# with col1:
#     amount = st.number_input("Transaction Amount ($)", min_value=0.0,
#                               max_value=100000.0, value=100.0, step=1.0)
#     time   = st.number_input("Time (seconds since first transaction)",
#                               min_value=0.0, value=50000.0, step=100.0)

# with col2:
#     st.write("Principal Component Features (V1 - V10)")
#     v_vals = {}
#     for i in range(1, 11):
#         v_vals[f"V{i}"] = st.number_input(
#             f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

# st.write("Additional PCA Features (V11 - V28)")
# cols = st.columns(3)
# for i in range(11, 29):
#     col_idx = (i - 11) % 3
#     with cols[col_idx]:
#         v_vals[f"V{i}"] = st.number_input(
#             f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

# # ── Prediction ────────────────────────────────────────────────────────────────
# if st.button("Predict", type="primary"):

#     # Build feature vector in the same order as training
#     from sklearn.preprocessing import StandardScaler
#     import pandas as pd

#     raw = {f"V{i}": v_vals[f"V{i}"] for i in range(1, 29)}

#     # Scale Amount and Time the same way as training
#     scaler_amt = StandardScaler()
#     amount_scaled = (amount - 88.35) / 250.12    # approx dataset mean/std
#     time_scaled   = (time - 94813.0) / 47488.0

#     raw["Amount_scaled"] = amount_scaled
#     raw["Time_scaled"]   = time_scaled

#     # Add engineered features
#     raw["V14_V17"] = raw["V14"] * raw["V17"]
#     raw["V12_V10"] = raw["V12"] * raw["V10"]

#     input_df  = pd.DataFrame([raw])

#     # Apply selector (select same features used during training)
#     try:
#         input_sel = selector.transform(input_df)
#         prob      = model.predict_proba(input_sel)[0][1]
#         pred      = int(prob > 0.5)
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         st.stop()

#     st.divider()
#     st.subheader("Prediction Result")

#     if pred == 1:
#         st.error(f"FRAUDULENT TRANSACTION  (Confidence: {prob * 100:.2f}%)")
#         st.write("This transaction shows characteristics consistent with fraud."
#                  " Consider blocking or flagging for manual review.")
#     else:
#         st.success(f"LEGITIMATE TRANSACTION  (Fraud Probability: {prob * 100:.2f}%)")
#         st.write("This transaction appears to be legitimate.")

#     st.metric("Fraud Probability", f"{prob * 100:.2f}%")

#     # Simple probability bar
#     st.progress(float(prob))
#     st.caption("0% = Definitely Legit  |  100% = Definitely Fraud")

# # ── Sidebar info ──────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.header("About")
#     st.write("""
#     This app uses a **Tuned Random Forest** model trained on the
#     Kaggle Credit Card Fraud Detection dataset.

#     **Model Performance:**
#     - High AUC-ROC score
#     - Trained with SMOTE to handle class imbalance
#     - Features selected using SelectKBest

#     **Dataset:**
#     284,807 transactions | 492 frauds (0.17%)
#     """)
#     st.write("Built for ML InnovateX Hackathon")
