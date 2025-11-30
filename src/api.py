# קובץ: src/api.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
from torchvision import models, transforms
from PIL import Image
import io

# שימו לב: אין כאן import streamlit!

# 1. יצירת האפליקציה
app = FastAPI()

# 2. הגדרת המודל
class_names = ['fox', 'gazelle']
device = torch.device("cpu")

model = models.resnet50(pretrained=False)  # טעינת המבנה
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))

# טעינת המשקולות (ודא שהנתיב נכון!)
try:
    model.load_state_dict(torch.load("../models/species_model.pt", map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 3. הכנת התמונה
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.get("/")
def read_root():
    return {"status": "Server is running"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # קריאת הקובץ
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # עיבוד וחיזוי
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)

    confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx].item()
    label = class_names[predicted_idx]

    # כאן בהמשך תוסיף את השמירה לדאטה-בייס

    return {
        "species": label,
        "confidence": f"{confidence:.2f}"
    }


# 4. הבלוק שמאפשר להריץ עם כפתור Run
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)