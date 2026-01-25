import google.generativeai as genai
import os

# ⚠️ שים כאן את ה-API Key שלך
GOOGLE_API_KEY = "AIzaSyDLerzOUatS6qLLjKOeBXSDY_gTKEHufvM"
genai.configure(api_key=GOOGLE_API_KEY)

print("🔍 בודק מודלים זמינים...")
try:
    for m in genai.list_models():
        # מחפש רק מודלים שיודעים לייצר טקסט (generateContent)
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")