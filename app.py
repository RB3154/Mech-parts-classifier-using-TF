import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models

# ─── Hyper‑params & metadata ────────────────────────────────────────────────
IMG_SIZE      = (300, 300)  # must match training
CLASS_NAMES   = ["bolt", "locatingpin", "nut", "washer"]
WEIGHTS_PATH  = "effnet_weights_final.weights.h5"

# ─── Model loader – cached so it only happens once per session ─────────────
@st.cache_resource(show_spinner="Warming up neural synapses…🧠")
def load_model():
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3)
    )
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = models.Model(inputs, outputs, name="EffNetB3_parts_classifier")
    model.load_weights(WEIGHTS_PATH)
    return model

model = load_model()

# ─── Page settings ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mechanical Parts Classifier",
    page_icon="🔧",
    layout="wide",
)

st.title("🔩🆚🔧  Mechanical Parts Image Classifier")
st.caption(
    "Upload a JPG/PNG of a mechanical part &rarr; get the predicted class and "
    "confidence. Powered by EfficientNet B3 + your fine‑tuned weights."
)

# ─── File uploader ─────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drag an image here or click to browse 👀",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    # Show the raw image
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Your uploaded image", use_column_width=False, width=300)

    # Pre‑process
    img_resized = image.resize(IMG_SIZE)
    img_np = np.array(img_resized).astype(np.float32)
    img_np = preprocess_input(img_np)
    img_tensor = np.expand_dims(img_np, axis=0)

    # Predict
    preds = model.predict(img_tensor, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    pred_label = CLASS_NAMES[top_idx]
    confidence = preds[top_idx] * 100

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🎯 Prediction")
        st.write(f"**{pred_label.capitalize()}**  \nConfidence: **{confidence:.2f}%**")

    with col2:
        st.subheader("🔍 Class probabilities")
        # Streamlit's built‑in bar_chart loves a tiny DataFrame‑like dict
        st.bar_chart({cls: float(preds[i] * 100) for i, cls in enumerate(CLASS_NAMES)})

    st.markdown("---")
    st.info("Need to classify another image? Just upload it above!")

# ─── Footer love ────────────────────────────────────────────────────────────
st.markdown(
    """
    <small>
    Model weights © 2025 RT • EfficientNet B3 backbone © Google.<br>
    App compiled with TensorFlow 2.19 & Streamlit 1.x.  Enjoy! ✨
    </small>
    """,
    unsafe_allow_html=True,
)
