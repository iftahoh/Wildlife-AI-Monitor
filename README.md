# Wildlife AI Monitor 🦌📷

An automated system for wildlife monitoring using Computer Vision and Deep Learning.
This project is designed to identify animal species from camera trap images and analyze their physical condition (Healthy/Injured) to aid conservation efforts.

## 👥 The Team
* Iftah Ohayon
* Alexay Laikov
* Talia Barzilai

## 🚀 Features
* **Species Classification:** Automatically identifies Israeli wildlife (Gazelles, Foxes, Boars, etc.) using Transfer Learning (ResNet50).
* **Health Analysis:** (In Progress) Detects signs of injury or malnutrition.
* **Live Dashboard:** A user-friendly web interface for uploading images and viewing real-time analytics.
* **Data Logging:** Automatically saves sighting history and confidence scores to a database.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **AI/ML:** PyTorch, Torchvision (ResNet50 / YOLOv8)
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Database:** SQLite (Development), PostgreSQL (Production)
* **Image Processing:** OpenCV, PIL

## 📂 Project Structure
```text
Wildlife-AI-Monitor/
├── data/                  # Raw images (Train/Val) - *Not in Git*
├── models/                # Trained model weights (.pt files)
├── src/
│   ├── api.py             # FastAPI backend server
│   ├── dashboard.py       # Streamlit frontend interface
│   ├── database.py        # Database management
│   └── train_model.py     # Training script (PyTorch)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation