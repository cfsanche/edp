# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:34:11 2018

@author: Camila
"""

import numpy as np
import cv2

# Load image and keep a copy
image = cv2.imread('C:\\Users\\Camila\\Documents\\EDP\\Illumination\\correction_test1.jpg')
orig_image = image.copy()
#ro,col = image.shape
#orig_image = cv2.resize(orig_image,(np.around(col/2).astype(int),np.around(ro/2).astype(int)))
cv2.imshow('Original Image', orig_image)
cv2.waitKey(0) 

# Grayscale and binarize
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours 
im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Iterate through each contour and compute the bounding rectangle
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)    
    cv2.imshow('Bounding Rectangle', orig_image)

cv2.waitKey(0) 
    
# Iterate through each contour and compute the approx contour
for c in contours:
    # Calculate accuracy as a percent of the contour perimeter
    accuracy = 0.0008* cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    cv2.imshow('Approx Poly DP', image)

cv2.imwrite('lesionsegment66.jpg',image)
cv2.waitKey(0)   
cv2.destroyAllWindows()