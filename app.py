import numpy as np
import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
with open('cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title and description
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🧬", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧬 Breast Cancer Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter tumor measurements to check if the tumor is <b>Benign</b> or <b>Malignant</b>.</p>", unsafe_allow_html=True)

st.markdown(
    """
    <style> 
    .stApp {
        background-color: #121212;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst"
]

# Initialize session state
if 'inputs' not in st.session_state:
    st.session_state.inputs = [0.0] * len(feature_names)

with st.sidebar:
    st.markdown("## ℹ️ About the App")
    st.write("""
    This app uses a trained machine learning model to detect whether a tumor is **Benign** or **Malignant**.
    
    ✅ Use sample data to test it  
    ⚠️ This tool is for educational/demo purposes only.
    """)
    st.markdown("---")
    st.markdown("👨‍💻 Developed during Internship at InternPe")

# Sample benign and malignant values
sample_benign_values = [
    12.0, 14.5, 78.0, 450.0, 0.08,        # mean values
    0.06, 0.03, 0.02, 0.18,               # mean continued
    0.30, 1.0, 2.0, 20.0, 0.01,           # standard error values
    0.01, 0.01, 0.01, 0.02,               # standard error continued
    14.0, 16.0, 90.0, 600.0, 0.09,        # worst values
    0.08, 0.05, 0.03, 0.20                # worst continued
]

sample_malignant_values = [
    20.0, 23.0, 132.0, 1400.0, 0.12,
    0.20, 0.25, 0.15, 0.30, 0.09,
    1.0, 1.5, 2.0, 25.0, 0.01,
    0.03, 0.04, 0.05, 0.03, 0.008,
    25.0, 28.0, 160.0, 1700.0, 0.15,
    0.30, 0.25
]

col_btn1, _ = st.columns([1, 5])
with col_btn1:
    if st.button("🧪 Use Sample Benign Data"):
        st.session_state.inputs = sample_benign_values
        st.toast("Benign sample_data loaded", icon="🟢")

    if st.button("🧪 Use Sample Malignant Data"):
        st.session_state.inputs = sample_malignant_values
        st.toast("Malignant sample_data loaded", icon="🔴")

st.markdown("---")
st.markdown("### 🧬 Enter Feature Values Below")

def render_feature_section(title, suffix):
    st.markdown(f"#### 🔹 {title}")
    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        if feature.endswith(suffix):
            with cols[i % 3]:
                st.session_state.inputs[i] = st.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    value=float(st.session_state.inputs[i]),
                    step=0.01,
                    format="%.2f",
                    key=feature
                )
    st.markdown("---")

render_feature_section("Mean Values", "_mean")
render_feature_section("Standard Error (SE) Values", "_se")
render_feature_section("Worst Case Values", "_worst")

# Predict Button
predict = st.button("🔍 Predict", use_container_width=True)

# Model Prediction
if predict:
    input_df = pd.DataFrame([st.session_state.inputs], columns=feature_names)

    # Check if all values are zero
    if np.all(input_df == 0):
        st.warning("⚠️ Please enter valid values. All-zero input may not give a meaningful prediction.")
    else:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][prediction]

        st.markdown("---")
        st.markdown("### 🧪 Prediction Result")

        if prediction == 0:
            st.success(f"✅ The tumor is **Benign** with a confidence of **{probability * 100:.2f}%**.")
            st.progress(probability)
            st.balloons()
        else:
            st.error(f"❌ The tumor is **Malignant** with a confidence of **{probability * 100:.2f}%**.")
            st.markdown(
                "<p style ='color : red; font-weight :bold;'>⚠️ Please consult a medical expert for further testing. "
                "This prediction is only for demo/educational purposes and should not be used for diagnosis.</p>",
                unsafe_allow_html = True
            )
            st.progress(probability)
