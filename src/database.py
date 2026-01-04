import sqlite3
from datetime import datetime

DB_NAME = "wildlife.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # הוספנו את quantity INTEGER
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
    print(f"Database {DB_NAME} initialized successfully.")


# הוספנו את הפרמטר quantity גם כאן
def add_sighting(filename, species, quantity, confidence, condition="Unknown"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO sightings (filename, species, quantity, confidence, condition)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, species, quantity, confidence, condition))

    conn.commit()
    sighting_id = cursor.lastrowid
    conn.close()
    return sighting_id


if __name__ == "__main__":
    init_db()