import sqlite3
import os
from datetime import datetime

# --- תיקון נתיב הדאטה-בייס ---
# זה מבטיח שהקובץ יישמר תמיד באותה תיקייה, לא משנה מאיפה מריצים את הקוד
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "wildlife.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            species TEXT,
            quantity INTEGER, 
            confidence REAL,
            condition TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")


def add_sighting(filename, species, quantity, confidence, condition="Unknown"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO sightings (filename, species, quantity, confidence, condition)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, species, quantity, confidence, condition))

    conn.commit()
    sighting_id = cursor.lastrowid
    conn.close()
    return sighting_id


# --- הפונקציה שהייתה חסרה לך ---
def get_all_sightings():
    """שולף את כל ההיסטוריה מהטבלה, מהחדש לישן"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sightings ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows


def run_custom_query(sql_query):
    """מריץ שאילתת SQL שנוצרה על ידי ה-AI ומחזיר את התוצאות"""
    try:
        conn = sqlite3.connect(DB_PATH)  # וודא ש-DB_PATH מוגדר כמו בתיקונים הקודמים
        cursor = conn.cursor()

        cursor.execute(sql_query)

        # אם זו שאילתת שליפה (SELECT) - נחזיר נתונים
        if sql_query.strip().upper().startswith("SELECT"):
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            return rows, columns, None  # מחזיר: שורות, שמות עמודות, שגיאה (ריקה)

        # אם זו פקודה אחרת (למשל סתם תרגול)
        conn.commit()
        conn.close()
        return None, None, "הפקודה בוצעה בהצלחה (אין נתונים להצגה)"

    except Exception as e:
        return None, None, f"SQL Error: {str(e)}"


if __name__ == "__main__":
    init_db()