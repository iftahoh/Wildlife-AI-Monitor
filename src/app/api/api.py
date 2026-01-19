from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# הוספת התיקייה הראשית לנתיב החיפוש כדי שה-import יעבוד
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.app.DB.database import add_sighting

app = FastAPI()

# --- התיקון לנתיב המודל ---
# 1. מציאת התיקייה שבה נמצא api.py (שהיא src/app/api)
CURRENT_DIR = Path(__file__).resolve().parent

# 2. עליה שתי קומות למעלה כדי להגיע ל-src (src/app/api -> src/app -> src)
SRC_DIR = CURRENT_DIR.parent.parent

# 3. בניית הנתיב לתיקיית models שנמצאת בתוך src
MODEL_PATH = SRC_DIR / "models" / "best.pt"

print(f"Looking for model at: {MODEL_PATH}")

if not MODEL_PATH.exists():
    print(f"❌ Error: Model not found at {MODEL_PATH}")
    # טוען מודל ברירת מחדל רק אם לא מוצא את המותאם
    model = YOLO("yolov8n.pt")
else:
    model = YOLO(MODEL_PATH)
    print("✅ Custom model loaded successfully!")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # קריאת התמונה
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # זיהוי
    results = model(img)

    # ספירת כמות וזיהוי סוג
    quantity = len(results[0].boxes)
    if quantity > 0:
        first_cls_id = int(results[0].boxes.cls[0].item())
        species_name = model.names[first_cls_id]
        confidence = float(results[0].boxes.conf[0].item())

        # שמירה לדאטה-בייס
        try:
            add_sighting(
                filename=file.filename,
                species=species_name,
                quantity=quantity,
                confidence=confidence,
                condition="Pending"
            )
            print(f"Saved to DB: {species_name} (x{quantity})")
        except Exception as e:
            print(f"Error saving to DB: {e}")

    # החזרת תמונה מסומנת
    annotated_img = results[0].plot()
    success, encoded_image = cv2.imencode('.jpg', annotated_img)
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
