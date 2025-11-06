import streamlit as st
from ..src.predict import load_model, predict_image

st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("🖼️ CIFAR-10 Image Classifier (PyTorch)")

@st.cache_resource
def _load():
    return load_model("outputs/best_model.pt")
model, device = _load()

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded:
    st.image(uploaded, caption="Uploaded", use_column_width=True)
    with open(".tmp_upload.png","wb") as f:
        f.write(uploaded.read())
    label, prob = predict_image(".tmp_upload.png", model, device)
    st.success(f"Prediction: **{label}**  (confidence: {prob:.3f})")
