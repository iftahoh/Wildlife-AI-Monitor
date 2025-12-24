from fastapi import FastAPI, UploadFile, File
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io
from database import add_sighting

app = FastAPI()

# טעינת מודל YOLO
# ודא שהקובץ yolov8n.pt נמצא בתיקיית models
model = YOLO("../models/yolov8n.pt")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # קריאת התמונה
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # ביצוע חיזוי עם YOLO
    results = model(image)

    # שליפת התוצאה הראשונה (אם יש זיהוי)
    label = "Unknown"
    confidence = 0.0

    # בודקים אם יש זיהויים (Boxes)
    if results[0].boxes:
        # לוקחים את הזיהוי עם הביטחון הכי גבוה
        box = results[0].boxes[0]
        class_id = int(box.cls)
        label = results[0].names[class_id]
        confidence = float(box.conf)

    # שמירה לדאטה-בייס
    add_sighting(
        filename=file.filename,
        species=label,
        confidence=confidence,
        condition="Pending"
    )

    return {
        "species": label,
        "confidence": f"{confidence:.2f}"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)