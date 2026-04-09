import os


def clean_unlabeled_images(images_dir, labels_dir):
    """Delete image files that have no matching YOLO label file.

    For each .jpg / .jpeg / .png in `images_dir`, the function looks for a
    corresponding .txt file in `labels_dir` with the same stem.  Images
    without a label are removed from disk to keep the dataset clean.

    Args:
        images_dir (str): Path to the folder containing image files.
        labels_dir (str): Path to the folder containing YOLO .txt label files.
    """
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
    # Resolve the YOLO data folder relative to this script's location
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    base_dir = os.path.join(project_root, "data", "YOLO")

    images_train = os.path.join(base_dir, "images", "train")
    labels_train = os.path.join(base_dir, "labels", "train")

    clean_unlabeled_images(images_train, labels_train)