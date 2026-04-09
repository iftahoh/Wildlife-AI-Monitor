import os
from ultralytics import YOLO

def main():
    """Fine-tune YOLOv8n on the project's wildlife dataset.

    Loads the base yolov8n.pt weights and trains for 50 epochs at 640px
    resolution using the dataset defined in data.yaml (located at the project root).
    Training artefacts (weights, metrics, plots) are saved by Ultralytics
    to runs/detect/train/ by default.

    Run from any directory — the data.yaml path is resolved relative to this file.
    """
    # Resolve data.yaml relative to this file so training works from any cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_yaml = os.path.join(project_root, "data.yaml")

    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml, epochs=50, imgsz=640)

if __name__ == "__main__":
    main()