import cv2
import os


def extract_frames_from_folder(source_folder, dest_folder, interval=30):
    """
    :param source_folder: איפה הסרטונים יושבים
    :param dest_folder: איפה לשמור את התמונות
    :param interval: כל כמה פריימים לשמור תמונה (30 = בערך תמונה אחת לשניה)
    """

    # יצירת תיקיית היעד אם לא קיימת
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # מעבר על כל הקבצים בתיקייה
    files = os.listdir(source_folder)
    video_count = 0
    total_images = 0

    print(f"Starting to process videos in {source_folder}...")

    for file_name in files:
        if file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".mts", ".asf", ".wmv", ".mpg")):
            video_path = os.path.join(source_folder, file_name)
            cap = cv2.VideoCapture(video_path)

            frame_count = 0
            saved_count = 0
            video_name = os.path.splitext(file_name)[0]

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # שמירה רק אם עברנו את המרווח שקבענו (למשל כל 30 פריימים)
                if frame_count % interval == 0:
                    # יצירת שם ייחודי לתמונה: videoName_frameNumber.jpg
                    out_name = f"{video_name}_f{frame_count}.jpg"
                    out_path = os.path.join(dest_folder, out_name)

                    cv2.imwrite(out_path, frame)
                    saved_count += 1
                    total_images += 1

                frame_count += 1

            cap.release()
            video_count += 1
            print(f"Finished {file_name}: Extracted {saved_count} images.")

    print(f"Done! Processed {video_count} videos, created {total_images} images.")


if __name__ == "__main__":
    # --- הגדרות ---
    # שנה את הנתיבים האלה לפי המחשב שלך!

    # איפה הסרטונים הגולמיים שקיבלתם
    videos_path = r"C:\Users\iftah\Desktop\hyena"

    # לאן לשפוך את התמונות המוכנות (לתוך תיקיית האימון של הפרויקט)
    images_output_path = r"../data/train/hyena"

    # הפעלה
    extract_frames_from_folder(videos_path, images_output_path, interval=30)