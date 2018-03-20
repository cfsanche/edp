# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:16:51 2018

@author: Camila
"""
import cv2

img_num = ['463','074','164','299','310']
thresh = []

for i in range(len(img_num)):
   
    img = cv2.imread("C:\\Users\\Camila\\Documents\\EDP\\Hair Removal\\zISIC_0000"+img_num[i]+"_hair.jpg",0)
    ro,col = img.shape
    hair = 0
    for j in range(col):
        for k in range(ro):
            if img[k][j] == 0:
                hair = hair+1
                
    thresh.append(hair)