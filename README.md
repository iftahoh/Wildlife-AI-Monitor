# Wildlife AI Monitor 🦌📷

An automated system for wildlife monitoring using Computer Vision and Deep Learning.  
This project analyzes camera-trap images to detect wildlife species, count individuals, and support future analysis of physical condition (Healthy / Injured) in order to aid conservation efforts.

## 👥 The Team
* Iftah Ohayon  
* Alexey Laikov  
* Talia Barzilai  

## 🚀 Features
* **Object Detection (YOLOv8):** Detects and localizes multiple animals per image using bounding boxes.
* **Species Classification (Baseline):** Identifies Israeli wildlife (Gazelles, Foxes, Boars, etc.) using Transfer Learning (ResNet50).
* **Health Analysis:** (In Progress) Planned detection of injury or malnutrition based on cropped detections.
* **Live Dashboard:** A user-friendly Streamlit interface for uploading images and viewing predictions.
* **Data Logging:** Automatically saves detection results and confidence scores to a database.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **AI / ML:** PyTorch, Torchvision, Ultralytics YOLOv8
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Database:** SQLite (Development), PostgreSQL (Planned)
* **Image Processing:** OpenCV, PIL

## 📂 Project Structure
```text
Wildlife-AI-Monitor/
├── data/
│   ├── YOLO/                  # YOLO-formatted dataset (generated locally)
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── raw/                   # Raw images and XML annotations (Not in Git)
│
├── models/                    # Trained model weights (.pt files)
│
├── src/
│   ├── api.py                 # FastAPI backend server
│   ├── dashboard.py           # Streamlit frontend interface
│   ├── database.py            # Database management
│   ├── train_model.py         # Baseline ResNet training script
│   ├── xml_to_yolo.py         # XML → YOLO annotation conversion
│   └── video_to_images.py     # Video to frames extraction
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## ⚙️ Environment Setup
python -m venv .venv 

.\.venv\Scripts\activate

pip install -r requirements.txt
