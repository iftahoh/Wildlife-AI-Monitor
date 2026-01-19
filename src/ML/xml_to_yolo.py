import os
import xml.etree.ElementTree as ET

# סדר הקלאסים חייב להיות קבוע אצל כל הצוות
CLASSES = ['fox', 'gazelle', 'hyena', 'porcupine', 'rabbit']

# מיפוי שמות "מלוכלכים" מה-XML לשם מחלקה תקני
ALIASES = {
    "fox1": "fox",
    "fox2": "fox",
    "fox3": "fox",
    # תוסיף כאן אם יש לכם gazelle1 וכו'
}

def voc_to_yolo_bbox(w, h, xmin, ymin, xmax, ymax):
    # center x/y, width/height
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    bbox_w = (xmax - xmin)
    bbox_h = (ymax - ymin)

    # normalize
    return (x_center / w, y_center / h, bbox_w / w, bbox_h / h)

def normalize_class(name: str) -> str:
    name = name.strip()
    return ALIASES.get(name, name)

def convert_xml_file(xml_path: str, out_txt_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls_name = normalize_class(obj.find("name").text)

        if cls_name not in CLASSES:
            # מדלגים על מחלקות לא מוכרות כדי לא לשבור אימון
            continue

        cls_id = CLASSES.index(cls_name)

        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))

        x, y, bw, bh = voc_to_yolo_bbox(w, h, xmin, ymin, xmax, ymax)
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

    # אם אין אובייקטים תקינים - אפשר לייצר קובץ ריק או לא לייצר בכלל
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def batch_convert(xml_dir: str, labels_out_dir: str):
    for name in os.listdir(xml_dir):
        if not name.lower().endswith(".xml"):
            continue
        xml_path = os.path.join(xml_dir, name)
        txt_name = os.path.splitext(name)[0] + ".txt"
        out_txt_path = os.path.join(labels_out_dir, txt_name)
        convert_xml_file(xml_path, out_txt_path)


if __name__ == "__main__":
    # מוצא את התיקייה שבה הקובץ הנוכחי (xml_to_yolo.py) נמצא
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # הולך צעד אחד אחורה לתיקייה הראשית של הפרויקט (יוצא מ-src)
    project_root = os.path.dirname(current_script_dir)

    # בונה את הנתיב המלא לתיקיית הנתונים בצורה בטוחה
    xml_dir = os.path.join(project_root, 'data', 'train', 'gazelle', 'Recgazelle')
    labels_out_dir = os.path.join(project_root, 'data', 'YOLO', 'labels', 'train')

    print(f"Looking for XMLs in: {xml_dir}")

    if not os.path.exists(xml_dir):
        print(f"Error: The folder '{xml_dir}' does not exist!")
        print("Please check if the folder name is correct (capital letters matter) and that it contains the XML files.")
    else:
        batch_convert(xml_dir, labels_out_dir)
        print("Done.")

