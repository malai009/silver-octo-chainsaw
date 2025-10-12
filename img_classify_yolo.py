#YOLO classification is used to just tell what are the objects present in an image
#but not the location or shape of the object
#-# : copy pasted from https://docs.ultralytics.com/tasks/classify/#train
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training) #-#

# Train the model
results = model.train(data=r"C:\Users\Asus\OneDrive\Documents\OpenCV\YOLO_Class\image_class_YOLO", epochs=20, imgsz=64) #-#

##the below line is to be pasted in terminal
#yolo classify train data="C:\Users\Asus\OneDrive\Documents\OpenCV\YOLO_Class\image_class_YOLO" model=yolo11n-cls.pt epochs=1 imgsz=64

## find the train sub highest number in the directry, inside which a CSV file and YAML file.

## at the end of each epoch you get a trained model that you can absolutely use

## the weight file inside the train contains the best and last training model of all the epochs
## ususally the last model has high accuracy but the best model has the highest accuracy.
## but the last model is has been through more training

