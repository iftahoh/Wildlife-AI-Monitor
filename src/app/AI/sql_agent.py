import os
from google import genai
from dotenv import load_dotenv

_SCHEMA = """
    Table name: sightings
    Columns:
    - id (INTEGER): Auto-incremented primary key
    - filename (TEXT): Name of the image file
    - species (TEXT): The animal name (e.g., 'gazelle', 'fox', 'boar')
    - quantity (INTEGER): Number of animals in the sighting
    - confidence (REAL): AI confidence level (0.0 to 1.0)
    - condition (TEXT): Health status of the animal ('healthy', 'injured', or 'N/A' when health analysis is disabled)
    - timestamp (DATETIME): Time of sighting (YYYY-MM-DD HH:MM:SS)
"""


def text_to_sql(user_question):
    """Convert a natural-language question into a SQLite query using Gemini.

    The function embeds the full sightings table schema in the prompt so the
    model understands the available columns.  Hebrew questions are supported —
    the model is instructed to translate the intent into SQL without including
    Hebrew characters in the output query.

    The Gemini client is created fresh on each call so that a newly written
    .env file is always picked up without requiring a server restart.

    Args:
        user_question (str): A free-text question about the sightings data,
                             in English or Hebrew.

    Returns:
        str: A SQLite SELECT query string, or an error message prefixed with
             "Error:" if the API call failed.
    """
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY is not set. Add it to your .env file."

    prompt = f"""
    You are an expert SQL assistant.
    Based on the following database schema:
    {_SCHEMA}

    Translate this question into a valid SQLite query:
    Question: "{user_question}"

    Rules:
    1. Return ONLY the SQL query. No markdown, no explanations.
    2. If the user asks in Hebrew, translate the intent to SQL logic.
    3. Ignore case sensitivity for species names (use LOWER() if needed).
    4. Do not include Hebrew in the SQL query.
    """

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        import re
        sql = re.sub(r"```[a-zA-Z]*", "", response.text).replace("```", "").strip()
        return sql
    except Exception as e:
        return f"Error: {e}"