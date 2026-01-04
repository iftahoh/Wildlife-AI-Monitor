import streamlit as st
import requests
from PIL import Image
import io

# כותרת האתר
st.title("מערכת זיהוי חיות בר 🦁")

# העלאת קובץ
uploaded_file = st.file_uploader("בחר תמונה...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # הצגת התמונה המקורית
    image = Image.open(uploaded_file)
    st.image(image, caption='תמונה מקורית', use_column_width=True)

    # כפתור לביצוע זיהוי
    if st.button('זהה חיה'):
        with st.spinner('מנתח תמונה...'):
            try:
                # שליחת התמונה לשרת
                files = {"file": uploaded_file.getvalue()}
                res = requests.post("http://127.0.0.1:8000/predict", files=files)

                if res.status_code == 200:
                    # הצגת התמונה שחזרה מהשרת (עם הריבועים)
                    st.success("הזיהוי הסתיים בהצלחה!")
                    st.image(res.content, caption='...', use_container_width=True)
                else:
                    st.error("שגיאה בחיבור לשרת")
            except Exception as e:
                st.error(f"שגיאה: {e}")