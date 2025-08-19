"""
Streamlit app to classify waste images using a saved ACO-based feature model.

Expect the pickle to be a dict with keys:
 - 'scaler' : sklearn scaler (e.g. StandardScaler)
 - 'selected_idx' : array-like of selected feature indices
 - 'model' : sklearn classifier (must implement .predict or .predict_proba and .classes_)

Place `aco_feature_model.pkl` in the same folder as this file (or update MODEL_PATH).
Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import joblib
import io
import time
from pathlib import Path

# TensorFlow/Keras for feature extraction
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------
# Config / constants
# -------------------------
MODEL_PATH = "aco_feature_model.pkl"  # change if your file has a different name
IMG_SIZE = (224, 224)

# -------------------------
# Helpers: load artifacts
# -------------------------
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_artifacts(model_path=MODEL_PATH):
    """
    Load pickle (scaler, selected_idx, model) and MobileNetV2 (feature extractor).
    Cached so it won't reload every interaction.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please upload it to the app folder.")
    artifacts = joblib.load(model_path)
    scaler = artifacts.get("scaler")
    selected_idx = np.asarray(artifacts.get("selected_idx"))
    clf = artifacts.get("model")

    # load MobileNetV2 (imagenet weights) as a frozen feature extractor
    feature_extractor = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    return scaler, selected_idx, clf, feature_extractor

def preprocess_pil_image(pil_img: Image.Image):
    """Prepare PIL image for MobileNetV2: RGB, resize, array, preprocess_input"""
    img = pil_img.convert("RGB")
    img = ImageOps.fit(img, IMG_SIZE, Image.ANTIALIAS)
    arr = img_to_array(img)  # height,width,ch
    arr = preprocess_input(arr)  # MobileNetV2 preprocessing
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

def predict_waste(pil_img: Image.Image, scaler, selected_idx, clf, extractor):
    """Full pipeline: extract features, scale, pick indices, predict"""
    X_img = preprocess_pil_image(pil_img)
    features = extractor.predict(X_img)  # shape (1, n_features)
    # scale
    if scaler is not None:
        X_scaled = scaler.transform(features)
    else:
        X_scaled = features
    # select features
    X_sel = X_scaled[:, selected_idx]
    # prediction
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_sel)[0]
        pred_class = clf.classes_[np.argmax(probs)]
        confidence = float(np.max(probs))
    else:
        pred_class = clf.predict(X_sel)[0]
        confidence = None
    return pred_class, confidence

# -------------------------
# Frontend / UI styling
# -------------------------
st.set_page_config(page_title="Waste Classifier", page_icon="🗑️", layout="centered")

# Simple CSS to make the UI more appealing
st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #f7fff7 0%, #f0fbff 100%);}
    .title {font-size:36px; font-weight:700; color:#0b6623;}
    .big-label {font-size:22px; font-weight:600;}
    .result {font-size:28px; font-weight:700; color:#0b6623;}
    .muted {color:#666666}
    .card {
        border-radius:16px;
        padding:18px;
        background: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<div class='title'>🗑️ Waste Classifier — Biodegradable vs Non-Biodegradable</div>", unsafe_allow_html=True)
st.markdown("Upload a photo of waste (jpg/png). The model will analyze and tell whether it's biodegradable or not.")

# Columns layout for upload + info
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Quick example images (optional): provide instructions
    st.markdown("<div class='muted'>Tip: Make sure the waste item is reasonably centered and well-lit for best results.</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>"
                "<h4 style='margin:4px 0 6px 0'>About</h4>"
                "<div class='muted'>This app uses a MobileNetV2 feature extractor + an ACO-selected feature set\n"
                "and a lightweight classifier. Upload an image to predict.</div>"
                "</div>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    # small info
    st.markdown("### Confidence threshold")
    conf_thresh = st.slider("Show as 'confident' only above", 0.0, 1.0, 0.60, 0.05)

# Load artifacts once (show helpful error if missing)
try:
    with st.spinner("Loading model and feature extractor..."):
        scaler, selected_idx, clf, extractor = load_artifacts(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# If user uploaded an image
if uploaded_file is not None:
    # display uploaded image
    image_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(image_bytes))
    st.image(pil_img, caption="Uploaded image", use_column_width=True)

    # action button
    if st.button("Classify image"):
        with st.spinner("Classifying..."):
            start = time.time()
            try:
                pred_class, confidence = predict_waste(pil_img, scaler, selected_idx, clf, extractor)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                raise

            elapsed = time.time() - start

        # Prepare label text & emoji mapping
        # If the saved model used numeric classes like 0/1, map to readable labels:
        # Prefer to detect 0/1, otherwise print class value directly.
        # Default mapping: 0 -> Biodegradable, 1 -> Non-Biodegradable
        label_map = {0: "Biodegradable", 1: "Non-Biodegradable"}

        if isinstance(pred_class, (np.ndarray, list)):
            pred_val = pred_class[0]
        else:
            pred_val = pred_class

        # Build displayable text
        if isinstance(pred_val, (int, np.integer)) and pred_val in label_map:
            pretty_label = label_map[int(pred_val)]
        else:
            # fallback: if model used strings already
            pretty_label = str(pred_val).replace("_", " ").title()

        # confidence handling
        if confidence is None:
            conf_text = "—"
        else:
            conf_text = f"{confidence*100:.1f}%"

        # show result prominently
        st.markdown("---")
        if confidence is not None and confidence >= conf_thresh:
            emoji = "✅" if "biodegrad" in pretty_label.lower() else "⚠️"
        else:
            emoji = "🤔"

        st.markdown(f"<div class='result'>{emoji}  {pretty_label}</div>", unsafe_allow_html=True)
        st.write(f"Confidence: **{conf_text}**")
        st.write(f"Inference time: {elapsed:.2f}s")

        # extra commentary
        if "biodegrad" in pretty_label.lower():
            st.success("This item is predicted to be biodegradable. Consider composting if appropriate.")
        else:
            st.warning("This item is predicted to be non-biodegradable. Consider recycling options if available.")

        # optional: show raw predicted probabilities if available
        if hasattr(clf, "predict_proba") and confidence is not None:
            probs = clf.predict_proba(extractor.predict(preprocess_pil_image(pil_img))[:, selected_idx])[0]
            # align classes to text
            class_names = []
            for c in clf.classes_:
                if isinstance(c, (int, np.integer)) and int(c) in label_map:
                    class_names.append(label_map[int(c)])
                else:
                    class_names.append(str(c).replace("_", " ").title())
            prob_table = {class_names[i]: f"{probs[i]*100:.1f}%" for i in range(len(prob_table:=class_names))}
            st.table({"Class": list(prob_table.keys()), "Probability": list(prob_table.values())})

else:
    # show an attractive placeholder if no file yet
    st.markdown(
        """
        <div style="padding:18px; border-radius:12px; background:linear-gradient(90deg,#ffffff,#f7fff7);">
            <h3>Try it out</h3>
            <p class="muted">Upload a waste image (plastic bottle, banana peel, paper, etc.) and click <b>Classify image</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown("---")
st.markdown("Built with ❤️ — drop an image and get a prediction. If you see odd predictions, check your dataset labels and model training.")

