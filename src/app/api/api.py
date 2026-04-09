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
import torch.nn.functional as F

# --- Feature Flag ---
# Set to True to re-enable health analysis once the model is reliable enough.
HEALTH_ENABLED = False

# --- Path setup ---
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(SRC_DIR.parent))

from src.app.DB.database import add_sighting

app = FastAPI()

# --- 1. Load YOLO model ---
YOLO_PATH = SRC_DIR / "models" / "best.pt"
print(f"🔍 Looking for YOLO model at: {YOLO_PATH}")

if not YOLO_PATH.exists():
    print("⚠️ Custom YOLO not found, using default yolov8n.pt")
    yolo_model = YOLO("yolov8n.pt")
else:
    yolo_model = YOLO(YOLO_PATH)
    print("✅ YOLO model loaded!")

# --- 2. Load health model (only when HEALTH_ENABLED = True) ---
# NOTE: Health analysis is currently disabled via HEALTH_ENABLED flag.
# The code below is kept intact so it can be re-enabled without any rewrites.
health_model = None
health_transforms = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if HEALTH_ENABLED:
    HEALTH_MODEL_PATH = SRC_DIR / "models" / "health_model.pt"
    print(f"🔍 Looking for Health model at: {HEALTH_MODEL_PATH}")

    _resnet = models.resnet50(weights=None)
    num_ftrs = _resnet.fc.in_features
    _resnet.fc = nn.Linear(num_ftrs, 2)

    if HEALTH_MODEL_PATH.exists():
        try:
            _resnet.load_state_dict(torch.load(HEALTH_MODEL_PATH, map_location=device))
            _resnet.to(device)
            _resnet.eval()
            health_model = _resnet
            print("✅ Health model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading health weights: {e}")
    else:
        print("❌ Health model file not found — health analysis will be skipped.")

    health_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
else:
    print("ℹ️ Health analysis disabled (HEALTH_ENABLED=False).")

HEALTH_CLASS_NAMES = ['healthy', 'injured']


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Detect animals in an uploaded image and return an annotated JPEG.

    Pipeline:
        1. Decode the uploaded image with OpenCV.
        2. Run YOLOv8 to detect all animals and draw bounding boxes.
        3. Select the detection with the highest confidence as the primary result.
        4. If HEALTH_ENABLED=True, run the ResNet50 health classifier on every crop.
        5. Save the detection to the SQLite database via add_sighting().
        6. Return the annotated image as a JPEG response.

    Args:
        file (UploadFile): Image file (JPEG or PNG) uploaded via multipart form.

    Returns:
        Response: A JPEG image with bounding boxes drawn by YOLO.
                  HTTP 200 on success.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = yolo_model(img)
    quantity = len(results[0].boxes)
    annotated_img = results[0].plot()

    # Pick the detection with the highest confidence score
    final_species = "Unknown"
    final_confidence = 0.0
    overall_health = "N/A"

    if quantity > 0:
        best_box = max(results[0].boxes, key=lambda b: float(b.conf[0].item()))
        cls_id = int(best_box.cls[0].item())
        final_species = yolo_model.names[cls_id]
        final_confidence = float(best_box.conf[0].item())

        # Health analysis — only runs when HEALTH_ENABLED = True
        if HEALTH_ENABLED and health_model is not None:
            overall_health = "healthy"
            for box in results[0].boxes:
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
                        probs = F.softmax(outputs, dim=1)
                        prob_healthy = probs[0][0].item() * 100
                        prob_injured = probs[0][1].item() * 100

                        print(f"\n📊 HEALTH for {yolo_model.names[int(box.cls[0].item())]}:")
                        print(f"   Healthy: {prob_healthy:.2f}%  Injured: {prob_injured:.2f}%")

                        # Alert at 4% injured probability (conservative threshold)
                        if prob_injured >= 4:
                            overall_health = "injured"
                            print(f"⚠️ Anomaly detected (Confidence: {prob_injured:.2f}%)")

            color = (0, 255, 0) if overall_health == "healthy" else (0, 0, 255)
            cv2.putText(annotated_img, f"Health: {overall_health}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)

        try:
            add_sighting(file.filename, final_species, quantity, final_confidence, overall_health)
        except Exception as e:
            print(f"❌ DB Error: {e}")

    success, encoded_image = cv2.imencode(".jpg", annotated_img)
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except SystemExit:
        pass