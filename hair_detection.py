# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:39:34 2018

@author: Camila
"""
import numpy as np
import cv2 
import scipy.ndimage
import gaussian_filters as gfilt
from show_images import show_images
import createMatchedFilterBank as mffilt

img = cv2.imread('correction_test3.jpg',1) # 0 = grayscale
# C:\\Users\\Camila\\Pictures\\Capstone\\ISIC_0009953 #correction_test
#cv2.imshow('image2',img)
#cv2.waitKey(0)
#img = cv2.blur(img, (2,2))

imgLAB = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
L,a,b = cv2.split(imgLAB)
img = L
#gaussian filter applied only to the luminance of the CIE LAB components
#rows,cols,dim = imgLAB.shape
#
#G = np.ones((rows,cols))
#for u in range(rows):
#    for v in range(cols):
#        G
ro,col = img.shape
#img = cv2.resize(img,(np.around(col/2).astype(int),np.around(ro/2).astype(int)))
# https://journals-scholarsportal-info.ezproxy.lib.ryerson.ca/pdf/02780062/v08i0003/263_dobviriutmf.xml
sigma =2
Len=15
#DOGfilt = scipy.ndimage.filters.gaussian_filter(L,sigma,order=1,truncate=6)

#result_show = cv2.resize(DOGfilt,(384,512))
#cv2.imshow('detection',DOGfilt)
#cv2.waitKey(0)
#cv2.imshow('original',L)

# http://funcvis.org/blog/?p=51 

gf = gfilt.gaussian_matched_filter_kernel(Len,sigma) # second div
fdog = gfilt.fdog_filter_kernel(Len, sigma) # first div

# visualize:
#show_images([gf, fdog], ["Gaussian", "FDOG"])

bank_gf = mffilt.createMatchedFilterBank(gf,n=12)
bank_fdog = mffilt.createMatchedFilterBank(fdog,n=18) #n=4 , default = 6

#print ("Gaussian")
#show_images(bank_gf)
#print ("FDOG")
#show_images(bank_fdog)

result_gf = mffilt.applyFilters(img,bank_gf)
result_fdog = mffilt.applyFilters(img,bank_fdog)
#result_fdog_single = cv2.filter2D(img, -1, fdog)

#final = cv2.merge((result_fdog,a,b))
#cv2.imshow('final',final)
#cv2.imshow('second div',result_gf)
#result_fdog = cv2.medianBlur(result_fdog,7)
cv2.imshow('first deriv gaussian',result_fdog)
#cv2.imshow('single second deriv',result_fdog_single)
cv2.imshow('original',img)

# threshold first deriv
thresh = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,5)
blur = cv2.GaussianBlur(thresh, (5,5), 0)
_, th3 = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('edges_first deriv',th3)

kernel = np.ones((1,1),np.uint8)
opening = cv2.morphologyEx(th3,cv2.MORPH_OPEN,kernel)
dilate = cv2.dilate(th3,kernel,iterations=10)
closing = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
cv2.imshow('final',opening)
cv2.waitKey(0)

cv2.destroyAllWindows()