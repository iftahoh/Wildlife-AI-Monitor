# קובץ: src/dashboard.py
import streamlit as st
import requests
from PIL import Image

# כתובת השרת שלנו
API_URL = "http://127.0.0.1:8000/predict"

st.title("Wildlife AI Monitor 🦌📷")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Analyze Image'):
        # הצגת ספינר בזמן שהשרת עובד
        with st.spinner('Analyzing...'):
            try:
                files = {"file": uploaded_file.getvalue()}
                res = requests.post(API_URL, files=files)

                if res.status_code == 200:
                    result = res.json()
                    st.success(f"Detected: {result['species']}")
                    st.info(f"Confidence: {result['confidence']}")
                else:
                    st.error("Error connecting to server")
            except Exception as e:
                st.error(f"Connection failed. Is api.py running? Error: {e}")