import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import zipfile





# Function to preprocess image: resizing and blurring
def preprocess_image(image, new_width, new_height):
    image_arr=np.array(image)
    blurred_image = cv2.GaussianBlur(image_arr, (5, 5), 0)
    resized_image = cv2.resize(blurred_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    img_array = resized_image / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def ModelExtract():
    zip_file_path =  'quantized_model.zip'
    extract_folder = 'Model/'
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)


@st.cache_resource
def load_model():
    ModelExtract()  # Extract once
    interpreter = tf.lite.Interpreter(model_path="Model/quantized_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

def predict(img_array, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions


# Function to display the prediction result
def display_prediction(predictions):
    labels = ['Myocardial Infarction', 'Abnormal Heartbeat', 'History of Myocardial Infarction', 'Normal Heartbeat']
    pred_idx = np.argmax(predictions)
    pred_label = labels[pred_idx]
    st.markdown(f"### Predicted Heart Condition: **{pred_label}**")
    st.markdown(f"#### Confidence: **{predictions[0][pred_idx]*100:.2f}%**")

# Set up the Streamlit page
st.set_page_config(page_title="ECG Heart Condition Classification", page_icon="❤️")


# Title and description
st.title("ECG Heart Condition Classification")
st.markdown("""
    This app analyzes your ECG image to classify heart conditions into four categories:
    - **Myocardial Infarction**
    - **Abnormal Heartbeat**
    - **History of Myocardial Infarction**
    - **Normal Heartbeat**
    
    Upload your ECG image and let the model classify it!
""")

# File uploader
uploaded_file = st.file_uploader("Upload an ECG Image", type=["jpg", "jpeg", "png", "bmp"])

# Create a colorful background
st.markdown(
    """
    <style>
    body {
        background-color: #f7f7f9;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }
    .stFileUploader {
        background-color: #007BFF;
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded ECG Image", use_container_width=True)
    if st.button("Analyze ECG"):
        try:
            image = Image.open(uploaded_file)
            img_array = preprocess_image(image,960,540)
            predictions = predict(img_array, interpreter)
            display_prediction(predictions)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.stop()
else:
    st.markdown("### Please upload an ECG image to get started.")
