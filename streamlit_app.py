import streamlit as st
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
import time

MODEL_URL = "https://drive.google.com/file/d/1xDW7a6jV2u50Zfokbe2kEcuCc3at5yHA/view?usp=sharing"
MODEL_PATH = "garbage-efficientnetv2s-1relulayer-2026_04_09-04_15.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()
#@st.cache_resource
#def load_model():
    #model_path = "garbage-efficientnetv2s-1relulayer-2026_04_09-04_15 (1).keras"

   # return tf.keras.models.load_model(
        #model_path,
        #compile=False
   # )
#model = load_model()

def preprocess_image(image):
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

    image = image.convert("RGB")
    image = image.resize((384, 384))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0

def start_over():
    st.session_state.uploader_key += 1
    st.session_state.camera_key += 1
    st.rerun()
st.title("The SortSmart Waste Detector")
st.subheader("to make your sorting an easier and more sustainable process")
option = st.radio("Select option:", ["Upload Image", "Use Camera"])
st.write("Make sure inputted image is clear and shows the entire object")
image = None

class_names = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash" ]


if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == "Use Camera":
    camera_image = st.camera_input(
        "Take a picture",
        key = f"camera_{st.session_state.uploader_key}"
    )
    if camera_image:
        image = Image.open(camera_image)
if image:
    st.session_state.processed = True
    st.image(image, caption="Selected Image", use_column_width=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Preprocessing image...")
    processed_image = preprocess_image(image)
    progress_bar.progress(20)

    status_text.text("Running model prediction...")
    predictions = model.predict(processed_image)
    progress_bar.progress(40)
    status_text.text("Running model prediction...")
    progress_bar.progress(60)
    status_text.text("Running model prediction...")
    progress_bar.progress(80)

    predicted_class = np.argmax(predictions)
    progress_bar.progress(100)
    status_text.text("Operation complete!")


    label = class_names[predicted_class]

    recycle = ["cardboard", "glass", "metal", "paper", "plastic"]
    compost = ["biological"]
    landfill = ["trash", "clothes", "shoes"]
    toxic = ["battery"]

    if label in recycle:
        category = "Recycle"
    elif label in compost:
        category = "Compost"
    elif label in landfill:
        category = "Landfill"
    elif label in toxic:
        category = "Toxic"
    else:
        category = "Unknown"


    st.success(f"Type of waste: {label}")
    st.success(f"Prediction: {category}")

    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    st.button("Classify another image", on_click=start_over)


