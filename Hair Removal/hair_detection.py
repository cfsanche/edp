# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:39:34 2018

Notes:
    1. fix image import and save name to variable

@author: Camila and Mani
"""
import numpy as np
import cv2 
import scipy.ndimage
import gaussian_filters as gfilt
from show_images import show_images
import createMatchedFilterBank as mffilt
from matplotlib import pyplot as plt

img = cv2.imread('C:\\Users\\Camila\\Documents\\EDP\\Illumination\\correction_test2.jpg',0) # 0 = grayscale
# C:\\Users\\Camila\\Pictures\\Capstone\\ISIC_0009953 #correction_test
#cv2.imshow('image2',img)
#cv2.waitKey(0)
#img = cv2.blur(img, (2,2))

#imgLAB = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
#L,a,b = cv2.split(imgLAB)
#img = L
#gaussian filter applied only to the luminance of the CIE LAB components
#rows,cols,dim = imgLAB.shape
#
#G = np.ones((rows,cols))
#for u in range(rows):
#    for v in range(cols):
#        G
ro,col = img.shape
img = cv2.resize(img,(np.around(col/2).astype(int),np.around(ro/2).astype(int)))
# https://journals-scholarsportal-info.ezproxy.lib.ryerson.ca/pdf/02780062/v08i0003/263_dobviriutmf.xml
sigma =2
Len=2
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

bank_gf = mffilt.createMatchedFilterBank(gf)
bank_fdog = mffilt.createMatchedFilterBank(fdog,n=18) #n=4 , default = 6

#print ("Gaussian")
#show_images(bank_gf)
#print ("FDOG")
#show_images(bank_fdog)

result_gf = mffilt.applyFilters(img,bank_gf)
result_fdog = mffilt.applyFilters(img,bank_fdog)
result_fdog_single = cv2.filter2D(img, -1, fdog)

#final = cv2.merge((result_fdog,a,b))
#cv2.imshow('final',final)
#cv2.imshow('second div',result_gf)
#result_fdog = cv2.medianBlur(result_fdog,7)
cv2.imshow('first deriv gaussian',result_gf)
#cv2.imshow('single second deriv',result_fdog_single)
cv2.imshow('original',img)

# threshold first deriv

thresh = cv2.adaptiveThreshold(result_gf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,8)
blur = cv2.GaussianBlur(thresh, (5,5), 0)

_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imgplot = plt.imshow(blur)
plt.show()
temp=cv2.Canny(result_gf,80,500)

cv2.destroyAllWindows()
median = cv2.medianBlur(th3,3)
cv2.imshow('th3',th3)
cv2.waitKey(0)

kernelo = np.ones((2,2),np.uint8)
#remove noise
opening = cv2.morphologyEx(th3,cv2.MORPH_OPEN,kernelo)

cv2.imwrite('hairdetect2.jpg',opening)
#cv2.imshow('final',closing)
#cv2.imshow('opening',opening)
#cv2.waitKey(0)

cv2.destroyAllWindows()