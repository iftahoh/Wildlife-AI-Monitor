import sqlite3
from datetime import datetime

# שם קובץ הנתונים
DB_NAME = "wildlife.db"


def init_db():
    """
    יוצר את הטבלה אם היא לא קיימת.
    מריצים את הפונקציה הזו פעם אחת בתחילת הפרויקט.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # יצירת טבלת תצפיות
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            species TEXT,
            confidence REAL,
            condition TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database {DB_NAME} initialized successfully.")


def add_sighting(filename, species, confidence, condition="Unknown"):
    """
    פונקציה לשמירת תצפית חדשה (נקרא לה מתוך ה-API)
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO sightings (filename, species, confidence, condition)
        VALUES (?, ?, ?, ?)
    ''', (filename, species, confidence, condition))

    conn.commit()
    sighting_id = cursor.lastrowid
    conn.close()
    return sighting_id


def get_all_sightings():
    """
    שליפת כל ההיסטוריה (עבור הדאשבורד)
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sightings ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    # הרצה ראשונית ליצירת הקובץ
    init_db()