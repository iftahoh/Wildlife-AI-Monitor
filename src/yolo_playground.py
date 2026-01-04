from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

# --- שינוי 1: נתיב למודל המאומן שלך ---
# הנחה: האימון נשמר בתיקיית runs בתוך הפרויקט. אם זה במקום אחר, תעדכן את הנתיב הזה.
# תחפש במחשב קובץ בשם best.pt ותעתיק את הנתיב שלו לכאן אם השורה למטה לא עובדת.
MODEL_PATH =  r"C:\Users\iftah\Desktop\All Files\לימודים\שנה ג\פרוייקט גמר\Wildlife-AI-Monitor\src\runs\detect\train3\weights\best.pt"

INPUT_IMAGE = SRC_DIR / "../data/train/gazelle/test.jpg"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_IMAGE = OUTPUT_DIR / "test_annotated_big_box.jpg"

# טעינת המודל המאומן במקום המודל הכללי
try:
    model = YOLO(MODEL_PATH)
except:
    print(f"Error: Could not find trained model at {MODEL_PATH}")
    print("Please update MODEL_PATH to point to your 'best.pt' file.")
    exit()


def expand_box(box, img_w, img_h, margin_ratio=0.3):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    x1_new = max(0, x1 - margin_ratio * w)
    y1_new = max(0, y1 - margin_ratio * h)
    x2_new = min(img_w, x2 + margin_ratio * w)
    y2_new = min(img_h, y2 + margin_ratio * h)

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def detect_and_draw_big_boxes():
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    img = cv2.imread(str(INPUT_IMAGE))
    img_h, img_w = img.shape[:2]

    # הרצת המודל
    results = model(str(INPUT_IMAGE))
    r = results[0]

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()

    print(f"Detected {len(boxes_xyxy)} objects.")

    annotated = img.copy()

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = expand_box(box, img_w, img_h, margin_ratio=0.4)

        # ציור הריבוע
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        conf = float(r.boxes.conf[i].item())
        cls_id = int(r.boxes.cls[i].item())

        # --- שינוי 2: שליפת שם החיה מתוך המודל ---
        class_name = model.names[cls_id]  # למשל: "Gazelle"

        label = f"{class_name}: {conf:.2f}"  # כותב את השם במקום רק ID

        cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMAGE), annotated)
    print(f"Saved annotated image to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    detect_and_draw_big_boxes()