import imaplib
import email
from email.header import decode_header
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = "imap.gmail.com"
API_URL = "http://127.0.0.1:8000/predict"

if not EMAIL_USER or not EMAIL_PASS:
    raise EnvironmentError("EMAIL_USER and EMAIL_PASS must be set in your .env file.")


def check_email():
    """Poll the Gmail inbox for unread messages and forward image attachments to the API.

    Connects to Gmail over IMAP, searches for UNSEEN messages, and for each
    message iterates over its parts looking for file attachments.  Any attachment
    found is POST-ed to the local FastAPI /predict endpoint as multipart form data.

    The function handles its own exceptions and prints status messages; it does
    not raise on network or IMAP errors so the polling loop can continue safely.
    """
    try:
        # התחברות לג'ימייל
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")

        # חיפוש אימיילים שלא נקראו
        status, messages = mail.search(None, "UNSEEN")
        email_ids = messages[0].split()

        if email_ids:
            print(f"found {len(email_ids)} new emails!")

        for e_id in email_ids:
            # קריאת האימייל
            res, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])

                    # חיפוש קבצים מצורפים (תמונות מהמצלמה)
                    for part in msg.walk():
                        if part.get_content_maintype() == 'multipart':
                            continue
                        if part.get('Content-Disposition') is None:
                            continue

                        file_name = part.get_filename()
                        if file_name:
                            print(f"Downloading image: {file_name}")
                            file_data = part.get_payload(decode=True)

                            # שליחה לשרת הזיהוי (API)
                            files = {'file': (file_name, file_data, 'image/jpeg')}
                            try:
                                response = requests.post(API_URL, files=files)
                                print(f"Sent to API. Status: {response.status_code}")
                            except Exception as e:
                                print(f"Error sending to API: {e}")

        mail.close()
        mail.logout()

    except Exception as e:
        print(f"Connection error: {e}")


if __name__ == "__main__":
    print("Email Bridge Started... Waiting for camera images.")
    while True:
        check_email()
        time.sleep(30)  # בודק כל 30 שניות