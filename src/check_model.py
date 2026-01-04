from ultralytics import YOLO
import os

# --- כאן נשים את הנתיב למודל שלך ---
# נסה למצוא את הנתיב הזה בדיוק במחשב שלך
model_path = r"C:\Users\iftah\Desktop\All Files\לימודים\שנה ג\פרוייקט גמר\Wildlife-AI-Monitor\src\runs\detect\train3\weights\best.pt"

print(f"Checking model at: {model_path}")

if not os.path.exists(model_path):
    print("❌ Error: The file does not exist at this path!")
    print("Please check where 'best.pt' is located on your computer.")
else:
    try:
        model = YOLO(model_path)
        print("\n✅ Model loaded successfully!")
        print("This model knows the following classes:")
        print(model.names)

        if 0 in model.names and model.names[0] == 'person':
            print("\n⚠️ WARNING: This looks like the DEFAULT YOLO model (it knows 'person', 'bicycle', etc.)")
            print("It seems your training didn't overwrite this file, or you are pointing to the wrong file.")
        else:
            print("\n🎉 GREAT! This is a custom model.")

    except Exception as e:
        print(f"Error loading model: {e}")