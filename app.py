import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------
# 🌿 Load the Trained CNN Model and Label Binarizer
# ---------------------------------------------------------
# Safely load model (pickle may wrap Keras model)
with open("cnn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_transform.pkl", "rb") as f:
    lb = pickle.load(f)

# ---------------------------------------------------------
# 🌱 Remedies for Detected Diseases
# ---------------------------------------------------------
REMEDIES = {
    "Pepper__bell___Bacterial_spot": "Use copper-based bactericides and avoid overhead irrigation.",
    "Pepper__bell___healthy": "Your pepper plant is healthy. Maintain good airflow and monitor regularly.",
    "Potato___Early_blight": "Apply fungicides like chlorothalonil and remove infected leaves.",
    "Potato___Late_blight": "Use resistant varieties and systemic fungicides such as metalaxyl.",
    "Potato___healthy": "Your potato plant is healthy. Ensure proper soil drainage and balanced fertilization.",
    "Tomato_Bacterial_spot": "Use copper sprays and avoid working with wet plants.",
    "Tomato_Early_blight": "Apply Mancozeb or chlorothalonil and prune lower leaves.",
    "Tomato_Late_blight": "Use fungicides like fluopicolide and destroy infected plants.",
    "Tomato_Leaf_Mold": "Improve air circulation and apply chlorothalonil or mancozeb.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves and apply fungicides.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticides or insecticidal soap; introduce natural predators.",
    "Tomato__Target_Spot": "Apply fungicides and remove infected foliage.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies and use virus-resistant varieties.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
    "Tomato_healthy": "Your tomato plant is healthy. Maintain proper watering and sunlight."
}

# ---------------------------------------------------------
# 🧠 Preprocess Image to Match Model Input
# ---------------------------------------------------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------------------------------------------
# 🔍 Predict Plant Disease
# ---------------------------------------------------------
def predict_disease(image):
    img_processed = preprocess_image(image)
    prediction = model.predict(img_processed)
    predicted_class = np.argmax(prediction, axis=1)
    label = lb.classes_[predicted_class[0]]
    confidence = np.max(prediction) * 100
    return label, confidence

# ---------------------------------------------------------
# 🎨 Streamlit Web App UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Plant Disease Detector 🌿", page_icon="🌿", layout="centered")

st.title("🌿 AI-Based Plant Disease Detector")
st.markdown("""
Upload one or more **leaf images** to detect plant diseases using a Convolutional Neural Network (CNN).  
This tool helps farmers identify diseases early and take corrective action.
""")

# File uploader
uploaded_files = st.file_uploader("📸 Upload leaf images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    report_data = []

    for uploaded in uploaded_files:
        image = Image.open(uploaded)
        st.image(image, caption=uploaded.name, use_container_width=True)

        with st.spinner(f"🔍 Analyzing {uploaded.name}..."):
            label, confidence = predict_disease(image)

        remedy = REMEDIES.get(label, "No remedy available for this class.")
        st.success(f"✅ Prediction: **{label}**")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.info(f"💡 **Suggested Remedy:** {remedy}")
        st.markdown("---")

        report_data.append({
            "Image": uploaded.name,
            "Prediction": label,
            "Confidence (%)": f"{confidence:.2f}",
            "Remedy": remedy
        })

    # Generate CSV report
    report_df = pd.DataFrame(report_data)
    csv = report_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="plant_disease_report.csv",
        mime="text/csv"
    )

    st.caption("Developed by Santhosh M | Minor Project | Department of Robotics & Automation")

else:
    st.warning("⬆️ Please upload one or more leaf images to get started.")
