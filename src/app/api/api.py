from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import sys
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F  # הוספנו את זה לחישוב אחוזים

# --- הגדרת נתיבים ---
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(SRC_DIR.parent))

from src.app.DB.database import add_sighting

app = FastAPI()

# --- 1. טעינת YOLO ---
YOLO_PATH = SRC_DIR / "models" / "best.pt"
print(f"🔍 Looking for YOLO model at: {YOLO_PATH}")

if not YOLO_PATH.exists():
    print(f"⚠️ Custom YOLO not found, using default yolov8n.pt")
    yolo_model = YOLO("yolov8n.pt")
else:
    yolo_model = YOLO(YOLO_PATH)
    print("✅ YOLO model loaded!")

# --- 2. טעינת מודל בריאות ---
HEALTH_MODEL_PATH = SRC_DIR / "models" / "health_model.pt"
print(f"🔍 Looking for Health model at: {HEALTH_MODEL_PATH}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# בניית המודל
health_model = models.resnet50(weights=None)
num_ftrs = health_model.fc.in_features
health_model.fc = nn.Linear(num_ftrs, 2)

if HEALTH_MODEL_PATH.exists():
    try:
        health_model.load_state_dict(torch.load(HEALTH_MODEL_PATH, map_location=device))
        health_model.to(device)
        health_model.eval()
        print("✅ Health model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading health weights: {e}")
        health_model = None
else:
    print("❌ Error: Health model file not found!")
    health_model = None

# הכנת התמונה
health_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# סדר הקטגוריות - חשוב מאוד! בדרך כלל אלפביתי
class_names = ['healthy', 'injured']


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. YOLO
    results = yolo_model(img)
    quantity = len(results[0].boxes)
    annotated_img = img.copy()

    final_species = "Unknown"
    final_confidence = 0.0
    final_health = "Unknown"

    if quantity > 0:
        box = results[0].boxes[0]
        cls_id = int(box.cls[0].item())
        final_species = yolo_model.names[cls_id]
        final_confidence = float(box.conf[0].item())

        # 2. Health Check
        if health_model is not None:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            h, w, _ = img.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            animal_crop = img[y1:y2, x1:x2]

            if animal_crop.size > 0:
                pil_img = Image.fromarray(cv2.cvtColor(animal_crop, cv2.COLOR_BGR2RGB))
                input_tensor = health_transforms(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = health_model(input_tensor)

                    # --- חישוב אחוזים (החלק החדש) ---
                    probs = F.softmax(outputs, dim=1)
                    prob_healthy = probs[0][0].item() * 100
                    prob_injured = probs[0][1].item() * 100

                    print(f"\n📊 ANALYSIS:")
                    print(f"   Healthy: {prob_healthy:.2f}%")
                    print(f"   Injured: {prob_injured:.2f}%")
                    # --------------------------------

                    _, preds = torch.max(outputs, 1)
                    predicted_idx = preds.item()

                    final_health = class_names[predicted_idx]
                    print(f"🩺 Final Verdict: {final_health}\n")

        # Visuals
        annotated_img = results[0].plot()

        # בחירת צבע: ירוק לבריא, אדום לפצוע
        color = (0, 255, 0) if final_health == 'healthy' else (0, 0, 255)

        cv2.putText(annotated_img, f"Health: {final_health}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Database
    if quantity > 0:
        try:
            add_sighting(file.filename, final_species, quantity, final_confidence, final_health)
        except Exception as e:
            print(f"❌ DB Error: {e}")

    success, encoded_image = cv2.imencode('.jpg', annotated_img)
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except SystemExit:
        pass