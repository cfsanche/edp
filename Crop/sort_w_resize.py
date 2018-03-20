# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:17:58 2018

@author: Camila
"""

import cv2
import glob
import os
import tensorflow as tf
from PIL import Image


fpaths = []
type_ = 'malignant'
sep_type = 'validation'
size = 336 #how many images
#dimension of output image
qual_val = 100
dim = 224 # for train 0:x, for validation x:end
datapath = "C:\\Users\\Camila\\Documents\\EDP\\Crop\\cropped_"+type_+"\\" 
fpaths = glob.glob(datapath+"*.jpg")
endpath = "C:\\Users\\Camila\\Documents\\EDP\\Dataset_test\\"+sep_type+"\\"+type_

for i in range(size+1,len(fpaths)): #(0,size):
    img = Image.open(fpaths[i]) # 0 = grayscale
    temp = fpaths[i]
    length = len(temp)
    pos = temp.rindex("\\")
    fname = temp[pos+1:length]
    pos = fname.index(".")
    fname = fname[0:pos]+"_norm"+str(dim)+".jpg"
    
    imager = img.resize((dim,dim),Image.LANCZOS)#img.resize((dim,dim),Image.ANTIALIAS)
    imagen = tf.image.per_image_standardization(img)
    imagen.save((os.path.join(endpath,fname),qual_val)
    #cv2.waitKey(0)