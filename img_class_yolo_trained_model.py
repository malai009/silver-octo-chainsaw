from ultralytics import YOLO
import numpy as np

model = YOLO(r".\runs\classify\train6\weights\last.pt")  # load a custom model

# Predict with the model
results = model(r"C:\Users\Asus\OneDrive\Documents\OpenCV\YOLO_Class\image_class_YOLO\train\cloudy\cloudy3.jpg")  # predict on an image

#print(results)
name_dict = results[0].names

probs = results[0].probs

print(name_dict)
print(probs.data) #the probility gives the tells about what the inserted image is more likely to be 

#print('the image is of a ', name_dict[np.argmax(probs)])

