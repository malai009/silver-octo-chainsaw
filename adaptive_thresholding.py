# Here we're applying specific threholds for specific regions in an img
# the algo moves through the image section by scection and computes a 
# threshold value for each section
import os
import cv2 as cv

img = cv2.imread(os.path.join('.', 'handwriten_letter.jpg'))
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#ret, thresh = cv2.adaptiveThreshold(gray_img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 
ret, thresh = cv2.adaptiveThreshold(gray_img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)

# we're not entering any threshold value here. 

cv2.imshow('frame', img)
cv2.imshow('frame_gray', gray_img)
cv2.imshow('frame_thresh', thresh)

cv2.waitKey(0)
