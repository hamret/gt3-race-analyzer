# __init__.py
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # small 버전
model = YOLO('yolov8m.pt')