import sqlite3
import os
from datetime import datetime

# --- תיקון נתיב הדאטה-בייס ---
# זה מבטיח שהקובץ יישמר תמיד באותה תיקייה, לא משנה מאיפה מריצים את הקוד
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "wildlife.db")


def init_db():
    """Create the sightings table if it does not already exist.

    The table stores one row per detection event with the following columns:
    id, filename, species, quantity, confidence, condition, timestamp.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
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
    """Insert a new detection record into the sightings table.

    Args:
        filename (str): Name of the source image file.
        species (str): Detected animal species (e.g. 'fox', 'gazelle').
        quantity (int): Number of animals detected in the image.
        confidence (float): YOLO confidence score for the primary detection (0.0–1.0).
        condition (str): Health status — 'healthy', 'injured', or 'N/A'. Defaults to 'Unknown'.

    Returns:
        int: The auto-generated id of the newly inserted row.
    """
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
    """Fetch all sighting records ordered from newest to oldest.

    Returns:
        list[tuple]: Each tuple contains (id, filename, species, quantity,
                     confidence, condition, timestamp).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sightings ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows


def run_custom_query(sql_query):
    """Execute an arbitrary SQL query and return the results.

    Intended for AI-generated SELECT queries from the SQL agent.
    Non-SELECT statements (e.g. UPDATE/DELETE) are executed but return no rows.

    Args:
        sql_query (str): A valid SQLite query string.

    Returns:
        tuple: (rows, columns, error) where:
            - rows (list[tuple] | None): Result rows for SELECT queries.
            - columns (list[str] | None): Column names for SELECT queries.
            - error (str | None): Error message if something went wrong, else None.
    """
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