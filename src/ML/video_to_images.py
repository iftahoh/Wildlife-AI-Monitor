import cv2
import os


def extract_frames_from_folder(source_folder, dest_folder, interval=30):
    """Extract frames from every video file in a folder and save them as JPEG images.

    Frames are sampled every `interval` frames (e.g. interval=30 keeps one
    frame per second at 30 fps).  Output filenames encode the animal type,
    video name, and original frame number for traceability.

    Args:
        source_folder (str): Path to a folder containing video files.
                             The folder's base name is used as the animal-type prefix.
        dest_folder (str): Path where extracted JPEG frames will be saved.
                           Created automatically if it does not exist.
        interval (int): Save one frame every this many frames. Defaults to 30.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    files = os.listdir(source_folder)
    total_images = 0

    print(f"Starting to process videos in {source_folder}...")

    for file_name in files:
        if file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".mts", ".asf", ".wmv", ".mpg")):
            video_path = os.path.join(source_folder, file_name)
            cap = cv2.VideoCapture(video_path)

            frame_count = 0
            saved_count = 0
            animal_type = os.path.basename(source_folder)
            video_name = os.path.splitext(file_name)[0]

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % interval == 0:
                    out_name = f"{animal_type}_{video_name}_f{frame_count}.jpg"
                    out_path = os.path.join(dest_folder, out_name)
                    cv2.imwrite(out_path, frame)
                    saved_count += 1
                    total_images += 1

                frame_count += 1

            cap.release()
            print(f"Finished {file_name}: Extracted {saved_count} images.")

    print(f"Done with folder {os.path.basename(source_folder)}! Created {total_images} images.")


if __name__ == "__main__":
    # --- הגדרות אוטומטיות ---

    # הפקודה הזו מוצאת לבד את תיקיית המסמכים שלך במחשב
    # היא מחפשת: C:\Users\YourName\Documents
    user_documents = os.path.join(os.path.expanduser("~"), "Documents")

    # כאן אנחנו מוסיפים את שם התיקייה שיצרת ("healthy" או "בריאה")
    # אם קראת לתיקייה בעברית "בריאה", תחליפי את "healthy" ב-"בריאה"
    base_videos_path = os.path.join(user_documents, "healthy")

    # לאן לשפוך את התמונות (לתוך הפרויקט)
    images_output_path = r"../../data/train/healthy"

    print(f"Looking for animal folders in: {base_videos_path}")

    if os.path.exists(base_videos_path):
        all_items = os.listdir(base_videos_path)

        found_any = False
        for item in all_items:
            item_path = os.path.join(base_videos_path, item)
            if os.path.isdir(item_path):
                print(f"\n--- Found animal folder: {item} ---")
                extract_frames_from_folder(item_path, images_output_path, interval=30)
                found_any = True

        if found_any:
            print("\n✅ All finished! Check your project data folder.")
        else:
            print("\n⚠️ Found the 'healthy' folder, but it has no animal folders inside.")
    else:
        print(f"❌ Error: Could not find the folder: {base_videos_path}")
        print("Please make sure you created a folder named 'healthy' inside your Documents.")