# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:39:34 2018

Notes:
    1. fix image import and save name to variable

@author: Camila and Mani
"""
import numpy as np
import cv2 
import gaussian_filters as gfilt
import createMatchedFilterBank as mffilt
import glob


sigma=2
Len=2
gf = gfilt.gaussian_matched_filter_kernel(Len,sigma) # second div
bank_gf = mffilt.createMatchedFilterBank(gf)

fpaths=[]
meta_path = "C:\\Users\\Camila\\Documents\\EDP\\DatasetSorted\\malignant\\"
fpaths=glob.glob(meta_path+"*.jpg")

for i in range (100):

    img = cv2.imread(fpaths[i],1)# 0 = grayscale
    ro,col,_ = img.shape
    img = cv2.resize(img,(np.around(col/2).astype(int),np.around(ro/2).astype(int)))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    y,_,_ = cv2.split(img)
    temp = fpaths[i]
    length = len(temp)
    pos = temp.rindex("\\")
    fname = temp[pos+1:length]
    pos = fname.find(".")
    fname = fname[0:pos]
    
    
    # https://journals-scholarsportal-info.ezproxy.lib.ryerson.ca/pdf/02780062/v08i0003/263_dobviriutmf.xml
    
    result_gf = mffilt.applyFilters(y,bank_gf)
    median = cv2.medianBlur(result_gf,3)
    thresh = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,8)
    blur = cv2.GaussianBlur(thresh, (5,5), 0)
    
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernelo = np.ones((2,2),np.uint8)
    #remove noise
    opening = cv2.morphologyEx(th3,cv2.MORPH_OPEN,kernelo)
    row,cols = opening.shape
    hair = 0
    for j in range(cols):
        for k in range(row):
            if opening[k][j] == 0:
                hair = hair+1
                
    if hair > 15000:
        cv2.imwrite(fname+'_remove.jpg',opening)
