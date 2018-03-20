# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 13:40:21 2018

@author: Camila
"""

import cv2
import glob

fpaths = []
type_ = 'benign'
datapath = "C:\\Users\\Camila\\Documents\\EDP\\DatasetSorted\\"+type_+"\\" 
fpaths = glob.glob(datapath+"*.jpg")

#endpath = "C:\\Users\\Camila\\Documents\\EDP\\Crop\\cropped_"+type_+"\\"

for i in range(len(fpaths)):
    img = cv2.imread(fpaths[i],-1) # 0 = grayscale
    temp = fpaths[i]
    length = len(temp)
    pos = temp.rindex("\\")
    fname = temp[pos+1:length]
    #    pos = fname.rindex("_")
    #    fname = fname[0:pos]
    #aISIC_0009874_seg.jpg    
    height, width,_ = img.shape
    if height < width:
        side_len = height
        diff = int((width - side_len)/2)
        imgc = img[:,diff:(width-diff)]
    else:
        side_len = width
        diff = int(height-side_len)
        imgc = img[diff:(height-diff),:]
        
    cv2.imwrite(fname+"_simplec",imgc)
        
    
    