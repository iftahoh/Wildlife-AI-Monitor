from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
from src.database import  add_sighting
import io

app = FastAPI()


MODEL_PATH = r"C:\Users\iftah\Desktop\All Files\לימודים\שנה ג\פרוייקט גמר\Wildlife-AI-Monitor\src\runs\detect\train4\weights\best.pt"

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = YOLO("yolov8n.pt")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. קריאת תמונה
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. זיהוי
    results = model(img)

    # --- חישוב הנתונים לשמירה ---
    # ספירת כמות הריבועים שזוהו
    quantity = len(results[0].boxes)

    # בדיקה איזו חיה זוהתה (לוקחים את הראשונה כנציגה, או "Unknown" אם לא מצא כלום)
    if quantity > 0:
        # שליפת ה-ID של המחלקה הראשונה שזוהתה
        first_cls_id = int(results[0].boxes.cls[0].item())
        species_name = model.names[first_cls_id]
        confidence = float(results[0].boxes.conf[0].item())
    else:
        species_name = "None"
        confidence = 0.0

    # 3. שמירה לדאטה-בייס עם הכמות החדשה!
    # (מוודאים שלא שומרים אם לא זוהה כלום, או שומרים כ-None לבחירתך)
    if quantity > 0:
        add_sighting(
            filename=file.filename,
            species=species_name,
            quantity=quantity,  # <--- הנה הכמות
            confidence=confidence,
            condition="Pending"
        )

    # 4. ציור והחזרת תמונה (כמו מקודם)
    annotated_img = results[0].plot()
    success, encoded_image = cv2.imencode('.jpg', annotated_img)
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)