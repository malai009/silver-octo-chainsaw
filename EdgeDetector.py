## you get the out line of the the subject in the image

import os
import cv2
import numpy as np

img = cv2.imread(os.path.join('.', 'CR7.webp'))

img_edge = cv2.Canny(img, 100, 200) #you get these values by trail and error

img_edge_d = cv2.dilate(img_edge, np.ones((3,3), dtype=np.int8)) #we're making the border thicker

img_edge_e = cv2.erode(img_edge_d,  np.ones((3,3), dtype=np.int8))#opposite of dilate

cv2.imshow('frame', img)
cv2.imshow('frame_edge', img_edge)
cv2.imshow('frame_edge_d', img_edge_d)
cv2.imshow('frame_edge_e', img_edge_e)
cv2.waitKey(0)