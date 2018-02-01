# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:12:24 2018

@author: Camila
"""
import cv2
from PIL import Image as Im
import numpy as np

#RGB Features
img = cv2.imread("C:\\Users\\Camila\\Documents\\EDP\\Illumination\\correction_test6.jpg",1)
segment = cv2.imread("C:\\Users\\Camila\\Documents\\EDP\\thresholding\\lesion_segment6.jpg",0)
_, segment = cv2.threshold(segment,127,255,cv2.THRESH_BINARY)
#cv2.imshow('img',l[[0]])

