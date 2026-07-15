"""
Surface Defect Detection — Streamlit web app
=============================================
Upload an image of a metal component surface; a CNN returns the defect type.

Deployed free on Streamlit Community Cloud. Needs three files in the same repo:
    streamlit_app.py     (this file)
    defect_model.keras   (the trained model)
    class_names.json     (label order)
and a requirements.txt.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Surface Defect Detection", page_icon="🔎", layout="centered")

DEFECT_INFO = {
    "crazing": "A fine network of hairline cracks. Usually thermal stress or over-rolling.",
    "inclusion": "Foreign material (slag, oxide) trapped in the surface layer, from impurities in the melt.",
    "patches": "Irregular discoloured regions where the surface texture differs from its surroundings.",
    "pitted_surface": "Clusters of small pits or cavities, typically from corrosion or descaling problems.",
    "rolled_in_scale": "Oxide scale pressed into the surface during rolling instead of being removed first.",
    "scratches": "Long, straight gouges from mechanical contact during handling or transport.",
}


# Loaded once and cached, so it doesn't reload on every interaction.
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("defect_model.keras")
    class_names = json.loads(Path("class_names.json").read_text())
    img_size = model.input_shape[1]          # read the size the model expects
    return model, class_names, img_size


model, class_names, IMG_SIZE = load_model_and_labels()

st.title("AI-Based Surface Defect Detection")
st.write(
    "A convolutional neural network trained on the NEU steel surface defect database. "
    "Upload an image of a component surface and it classifies the defect into one of six "
    "types: crazing, inclusion, patches, pitted surface, rolled-in scale, or scratches."
)

uploaded = st.file_uploader("Upload a component surface image", type=["bmp", "jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to run inspection. Sample NEU images are in the `examples/` folder of the repo.")
else:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input", use_container_width=True)

    # Same preprocessing as training: RGB, resize with BILINEAR to the model's size.
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype="float32")[None, ...]

    probs = model.predict(arr, verbose=0)[0]
    top = class_names[int(np.argmax(probs))]
    conf = float(np.max(probs))

    with col2:
        st.subheader(f"Detected: {top.replace('_', ' ').title()}")
        st.metric("Confidence", f"{conf * 100:.1f}%")
        st.caption(DEFECT_INFO.get(top, ""))
        if conf < 0.60:
            st.warning("Low confidence — on a real line this would be flagged for manual inspection.")

    st.write("**All class probabilities**")
    order = np.argsort(probs)[::-1]
    for i in order:
        st.write(f"{class_names[i].replace('_', ' ').title()}")
        st.progress(float(probs[i]))

st.divider()
st.caption(
    "Trained on the NEU Surface Defect Database (Northeastern University). "
    "MobileNetV2 fine-tuned; macro F1 ≈ 0.98 on a held-out test set. "
    "Proof of concept, not a production inspection system."
)
