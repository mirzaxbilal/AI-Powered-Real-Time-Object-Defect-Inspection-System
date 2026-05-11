from ultralytics import YOLO

# Load pretrained classification model
model = YOLO("yolov8s-cls.pt")

# Train
model.train(
    data="dataset",
    epochs=50,
    imgsz=640,
    batch=16,
    augment=True,
    project="runs/classify",
    name="bottle_classifier"
)