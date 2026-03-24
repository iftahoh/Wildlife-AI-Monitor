import google.generativeai as genai
import os

# ⚠️ הדבק כאן את המפתח שלך (או שמור כמשתנה סביבה)
GOOGLE_API_KEY = "AIzaSyAb0hUjtSMXfXZxz4tlFLhnFbKTjO2h-Kc"

genai.configure(api_key=GOOGLE_API_KEY)

def text_to_sql(user_question):
    """פונקציה שממירה טקסט חופשי לשאילתת SQL"""

    # תיאור הטבלה שלנו כדי שה-AI יבין
    schema = """
    Table name: sightings
    Columns:
    - filename (TEXT): Name of the image file
    - species (TEXT): The animal name (e.g., 'gazelle', 'fox', 'boar')
    - quantity (INTEGER): Number of animals in the sighting
    - confidence (REAL): AI confidence level (0.0 to 1.0)
    - timestamp (DATETIME): Time of sighting (YYYY-MM-DD HH:MM:SS)
    """

    prompt = f"""
    You are an expert SQL assistant.
    Based on the following database schema:
    {schema}
    
    Translate this question into a valid SQLite query:
    Question: "{user_question}"
    
    Rules:
    1. Return ONLY the SQL query. No markdown, no explanations.
    2. If the user asks in Hebrew, translate the intent to SQL logic.
    3. Ignore case sensitivity for species names (use LOWER() if needed).
    4. Do not include Hebrew in the SQL query.
    """

    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    try:
        response = model.generate_content(prompt)
        sql = response.text.strip().replace('```sql', '').replace('```', '')
        return sql
    except Exception as e:
        return f"Error: {e}"