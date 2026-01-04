from ultralytics import YOLO

def main():
    # טעינת מודל
    model = YOLO('yolov8n.pt')

    # הרצת האימון - שים לב לנתיב של ה-yaml
    # אנחנו מריצים מתוך תיקיית src, אז צריך לצאת החוצה פעם אחת (..)
    model.train(data='../data.yaml', epochs=50, imgsz=640)

if __name__ == '__main__':
    main()