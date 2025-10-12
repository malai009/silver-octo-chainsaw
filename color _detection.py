import cv2
import numpy as np
from PIL import Image # to convert mask from np to pillow
from color_detection_utl import get_limits

red = [0, 0, 255] # choosing green in BGR color space 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

# we're converting bgr to HSV(Hue, Saturation,Value) color space to 
# give the computer a range of colors to work with
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#setting the limits using the get_limit function
    lowerLimit, upperLimit = get_limits(red)
    #print(upperLimit)

# we're defining the range on HUE chart within which to take reference color from
#mask stores the location of the color we're interested in from the screen
    mask = cv2.inRange(hsv_img, lowerLimit, upperLimit) 

# now we're converting numpy to pillow

    mask_ = Image.fromarray(mask)
    bndng_box = mask_.getbbox() # drawing bounding box around the object of interest   
    print(bndng_box)

    if bndng_box is not None: 
        x1, y1, x2, y2 = bndng_box
        frame = cv2.rectangle(frame, (x1,y1), (x2,y2), [0, 255, 0], 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()

cv2.destroyAllWindows()






