import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

_api_key = os.getenv("GOOGLE_API_KEY")
if not _api_key:
    raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")

genai.configure(api_key=_api_key)

print("🔍 בודק מודלים זמינים...")
try:
    for m in genai.list_models():
        # מחפש רק מודלים שיודעים לייצר טקסט (generateContent)
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")