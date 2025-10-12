## Segmentation locates the object from an image in addition to identifying it
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=r"C:\Users\Asus\OneDrive\Documents\OpenCV\YOLO_seg\img_seg_yolo_path.yaml", epochs=1, imgsz=640)



