import os


def clean_unlabeled_images(images_dir, labels_dir):
    print(f"Cleaning images in: {images_dir}")

    removed_count = 0
    # עובר על כל הקבצים בתיקיית התמונות
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # בודק אם קיים קובץ txt תואם
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            # אם אין תיוג - מוחק את התמונה
            img_path = os.path.join(images_dir, img_name)
            os.remove(img_path)
            print(f"Removed unlabeled image: {img_name}")
            removed_count += 1

    print(f"\nDone. Removed {removed_count} images.")


if __name__ == "__main__":
    # נתיבים מלאים (כמו שהגדרנו קודם)
    base_dir = r"C:\Users\iftah\Desktop\All Files\לימודים\שנה ג\פרוייקט גמר\Wildlife-AI-Monitor\data\YOLO"
    images_train = os.path.join(base_dir, "images", "train")
    labels_train = os.path.join(base_dir, "labels", "train")

    clean_unlabeled_images(images_train, labels_train)