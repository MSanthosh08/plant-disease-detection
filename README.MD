# 🌿 AI-Based Plant Disease Detector

An intelligent deep learning web app built with **TensorFlow** and **Streamlit** that detects **plant leaf diseases** from images and suggests **remedies** to help farmers take corrective actions early.

---

## 🚀 Overview

This project uses a **Convolutional Neural Network (CNN)** trained on the **PlantVillage dataset** to identify common diseases in crops such as **tomato, potato, and pepper**.  
The app provides:
- Disease classification with confidence scores  
- Suggested remedies for each disease  
---

## 🧠 Features

✅ Upload one or more leaf images  
✅ Automatic preprocessing and disease detection  
✅ Confidence percentage for each prediction  
✅ Remedies and preventive actions   
✅ User-friendly **Streamlit** web interface  

---

## 🧩 Classes Detected

| Crop | Diseases Detected |
|------|--------------------|
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🧰 Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras**
- **Streamlit** (for web UI)
- **Pandas / NumPy**
- **Pillow**
- **Pickle** (for model and label encoding)

---

## 🧪 Model Details

- Model Type: **Convolutional Neural Network (CNN)**
- Input Size: `256 x 256 x 3`
- Output Classes: 15
- Framework: TensorFlow + Keras
- Trained on: PlantVillage dataset
- Optimizer: Adam
- Loss: Categorical Crossentropy

---

## ⚙️ Installation and Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/MSanthosh08/plant_disease_detector.git
cd plant_disease_detector
```

### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # For Linux/macOS
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Streamlit app
```bash
streamlit run app.py
```
## 📁 Project Structure
```bash 
plant_disease_detector/
│
├── app.py                     # Streamlit main app
├── cnn_model.pkl              # Trained CNN model (pickled)
├── label_transform.pkl        # Label encoder
└── README.md                  # Project documentation
```
## 🧾 Example Output

After uploading an image:

✅ Prediction: Tomato_Early_blight
Confidence: 97.24%
💡 Suggested Remedy: Apply Mancozeb or chlorothalonil and prune lower leaves.

## 🧠 Future Enhancements

🌱 Add more crop varieties and diseases

📈 Improve accuracy using transfer learning (e.g., EfficientNet, MobileNetV3)

🛰️ Integrate with IoT-based farm monitoring systems

📲 Deploy on web/mobile for real-time detection

👨‍💻 Developed By

Santhosh M
Minor Project — AI-Based Plant Disease Detection

📧 Contact: [m.santhosh0804@gmail.com]
🌐 GitHub: https://github.com/MSanthosh08

## 🪴 Acknowledgments

PlantVillage Dataset

TensorFlow

Streamlit

“Early detection leads to early protection.” 🌾
Empowering farmers with AI technology for sustainable agriculture.
