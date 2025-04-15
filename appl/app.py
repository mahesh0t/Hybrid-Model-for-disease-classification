import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mob
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_dense
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
import pickle

# Load ensemble model and scaler
model = joblib.load("ensemble_model.joblib")
print("✅ Model loaded successfully!")
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load pre-trained deep models
mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
densenet = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Disease class names
class_names = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Melanocytic Nevi",
    "Benign Keratosis",
    "Viral Infections"
]

# --------------------- Feature Extraction ------------------------
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import cv2
import numpy as np

def extract_lbp(image):
    gray = rgb2gray(image)  # [0, 1] float
    gray_uint8 = (gray * 255).astype(np.uint8)  # convert to 8-bit
    lbp = local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist.astype(np.float32)

def extract_glcm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy])

def extract_manual_features(image):
    # Assume image is already RGB and 224x224
    lbp_feat = extract_lbp(image)
    glcm_feat = extract_glcm(image)
    manual_feat = np.hstack([lbp_feat, glcm_feat])
    return scaler.transform([manual_feat])[0]

def extract_cnn_features(image):
    
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)

    mob_feat = mobilenet.predict(preprocess_mob(x), verbose=0)[0]
    dense_feat = densenet.predict(preprocess_dense(x), verbose=0)[0]
    return mob_feat, dense_feat

# ---------------------- UI Logic --------------------------
st.title("🩺 Skin Disease Classifier")
st.write("Upload a skin lesion image to diagnose one of the 6 supported conditions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption='Uploaded Image', use_container_width=True)

    st.write("\nExtracting features and predicting...")

    # Feature extraction
    img_resized = cv2.resize(image_rgb, (224, 224))
    manual_feat = extract_manual_features(img_resized)
    mob_feat, dense_feat = extract_cnn_features(img_resized)

    combined_features = np.hstack((manual_feat, mob_feat, dense_feat)).reshape(1, -1)

    # Prediction
    print("Testing model with dummy input...")
    print(model.predict(np.random.rand(1, 2318)))  # Change this!

    prediction = model.predict(combined_features)[0]
    predicted_class = class_names[prediction]

    st.success(f"🧾 Predicted Class: **{predicted_class}**")

    # Provide information
    if predicted_class == "Eczema":
        st.info("**Eczema:** Keep skin moisturized, avoid irritants. See a dermatologist if severe.")
    elif predicted_class == "Melanoma":
        st.error("**Melanoma:** Possibly dangerous. Consult a dermatologist or oncologist immediately.")
    elif predicted_class == "Atopic Dermatitis":
        st.info("**Atopic Dermatitis:** Moisturizers, avoid allergens. See a dermatologist.")
    elif predicted_class == "Melanocytic Nevi":
        st.info("**Melanocytic Nevi:** Usually benign. Monitor changes, consult a dermatologist.")
    elif predicted_class == "Benign Keratosis":
        st.info("**Benign Keratosis:** Non-cancerous. Remove only if bothersome. Dermatologist optional.")
    elif predicted_class == "Viral Infections":
        st.info("**Viral Infections (e.g., warts):** Topical treatments or cryotherapy. Consult a skin specialist.")

    st.markdown("---")
    st.markdown("*This is an AI-based tool. Please consult a medical professional for final diagnosis.*")
