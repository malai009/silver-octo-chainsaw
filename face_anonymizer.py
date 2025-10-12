import cv2
import os
import mediapipe as mp

## Read image
img_path = 'face_david.webp'

img = cv2.imread(img_path)

H, W, _ = img.shape

## Detect Faces using Mediapie library
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection( model_selection = 0, min_detection_confidence = 0.5) as face_detection:
# Arg1 is could be 0 or 1. 0 for faces within 2m from cam. 1 for faces within 5m from cam.
# Arg2 could be 0 to 1. 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # we want img to be in rgb for face detection
    img_out = face_detection.process(img_rgb) # output img after processing
    print(img_out.detections) # this is done to get the bounding box location of info of face

    if img_out is not None:  # to make sure the loop runs iff the image contains a human face, not an animal face

      for detection in img_out.detections: #these data have been copy passted from the output of print from termial
        location_data = detection.location_data
        bbx = location_data.relative_bounding_box

        x1, y1, w, h = bbx.xmin, bbx.ymin, bbx.width, bbx.height
        
        X1 = int(x1 * W)   # converting the coordinates of bbx into absolute values as they were previously 
        Y1 = int(y1 * H)   # relative
        W = int(w * W)
        H = int(h * H)

        cv2.rectangle(img, (X1,Y1), (X1+W,Y1+H), [0, 255, 0], 1 )

   

## Blur Faces
img_blur = cv2.blur(img[Y1:Y1+H, X1:X1+W, :], (80,80))
img[Y1:Y1+H, X1:X1+W, :] = img_blur

cv2.imshow('img', img)
#cv2.imshow('img_blr', img_blur)
cv2.waitKey(0)

## Save Image

cv2.imwrite(os.path.join('.', 'faceBlurOP.png'),img)