import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import altair as alt
import sys
import os

# --- 1. תיקון נתיבים (חייב להיות לפני ה-imports של src) ---
# עולים 3 רמות למעלה כדי להגיע לתיקיית השורש (Wildlife-AI-Monitor)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# --- 2. עכשיו אפשר לייבא בבטחה מהפרויקט ---
# שים לב: אם הקבצים האלו לא קיימים, תקבל שגיאה. וודא שיצרת אותם!
try:
    from src.app.DB.database import get_all_sightings, run_custom_query
    from src.app.AI.sql_agent import text_to_sql
except ImportError as e:
    st.error(f"שגיאת ייבוא קריטית: {e}")
    st.stop()

# --- הגדרות עמוד ---
st.set_page_config(page_title="Wildlife Monitor", layout="wide", page_icon="🦁")
st.title("🦁 Wildlife AI Monitor")

# --- יצירת טאבים (Tabs) ---
tab1, tab2, tab3, tab4 = st.tabs(["📸 זיהוי", "📋 היסטוריה", "📊 סטטיסטיקה", "🤖 שאל את הדאטה"])

# ==========================================
# טאב 1: זיהוי תמונה
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("העלאת תמונה")
        uploaded_file = st.file_uploader("בחר תמונה...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='תמונה מקורית', use_container_width=True)

            if st.button('הפעל זיהוי 🚀', type="primary"):
                with st.spinner('מנתח תמונה...'):
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format)
                    img_byte_arr = img_byte_arr.getvalue()

                    try:
                        files = {'file': (uploaded_file.name, img_byte_arr, 'image/jpeg')}
                        # וודא שהשרת (api.py) רץ ברקע!
                        res = requests.post("http://127.0.0.1:8000/predict", files=files)

                        if res.status_code == 200:
                            st.session_state['last_result'] = res.content
                            st.success("✅ זוהה בהצלחה!")
                        else:
                            st.error("❌ שגיאה מהשרת")
                    except Exception as e:
                        st.error(f"לא ניתן להתחבר לשרת: {e}")

    with col2:
        if 'last_result' in st.session_state:
            st.subheader("תוצאה")
            st.image(st.session_state['last_result'], use_container_width=True)

# ==========================================
# טאב 2: היסטוריה
# ==========================================
with tab2:
    st.subheader("תיעוד מפגשים")
    try:
        rows = get_all_sightings()
        if rows:
            df = pd.DataFrame(rows, columns=['ID', 'שם קובץ', 'סוג חיה', 'כמות', 'ביטחון', 'מצב', 'זמן'])
            df = df.drop(columns=['ID'])
            df['ביטחון'] = df['ביטחון'].apply(lambda x: f"{float(x):.1%}" if isinstance(x, (float, int)) else x)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("עדיין אין נתונים.")
    except Exception as e:
        st.error(f"שגיאה בטעינת נתונים: {e}")

# ==========================================
# טאב 3: סטטיסטיקה
# ==========================================
with tab3:
    st.subheader("ניתוח נתונים 📈")
    try:
        rows = get_all_sightings()
        if rows:
            df = pd.DataFrame(rows,
                              columns=['ID', 'filename', 'species', 'quantity', 'confidence', 'condition', 'timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # חישוב נכון: סכימת הכמויות
            total_animals = df.groupby('species')['quantity'].sum().reset_index()
            total_animals.columns = ['species', 'total_count']

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 🦌 התפלגות כמותית")
                pie = alt.Chart(total_animals).mark_arc(outerRadius=100).encode(
                    theta=alt.Theta("total_count", stack=True),
                    color=alt.Color("species"),
                    tooltip=["species", "total_count"]
                )
                st.altair_chart(pie, use_container_width=True)

            with col_b:
                st.markdown("### 🕒 שעות פעילות")
                df['hour'] = df['timestamp'].dt.hour
                hourly = df['hour'].value_counts().reset_index()
                hourly.columns = ['hour', 'count']

                bar = alt.Chart(hourly).mark_bar().encode(
                    x=alt.X("hour:O", title="שעה"),
                    y=alt.Y("count", title="אירועים"),
                    color=alt.value("#FFA500")
                )
                st.altair_chart(bar, use_container_width=True)
        else:
            st.info("אין מספיק נתונים.")
    except Exception as e:
        st.error(f"שגיאה בסטטיסטיקה: {e}")

# ==========================================
# טאב 4: צ'אט AI (החדש!)
# ==========================================
with tab4:
    st.header("🤖 צ'אט עם הנתונים")
    st.markdown("שאל שאלות חופשיות על המידע שנאסף.")

    question = st.text_input("מה תרצה לדעת?", placeholder="למשל: כמה צבאים ראינו השבוע?")

    if st.button("שאל את ה-AI ✨") and question:
        with st.spinner("ה-AI כותב שאילתה..."):
            try:
                # 1. יצירת SQL
                generated_sql = text_to_sql(question)
                st.code(generated_sql, language="sql")

                # 2. הרצה
                data, columns, error = run_custom_query(generated_sql)

                if error:
                    st.error(f"שגיאה: {error}")
                elif data:
                    df_res = pd.DataFrame(data, columns=columns)
                    st.dataframe(df_res, use_container_width=True)
                    st.success(f"נמצאו {len(data)} תוצאות")
                else:
                    st.warning("השאילתה רצה בהצלחה אך לא נמצאו תוצאות (אולי אין נתונים מתאימים?)")

            except Exception as e:
                st.error(f"תקלה כללית: {e}")