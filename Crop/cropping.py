# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:10:46 2018

@author: Camila
"""
import cv2
import numpy
import math
import glob

fpaths = []
type_ = 'malignant'
datapath = "C:\\Users\\Camila\\Documents\\EDP\\thresholding\\"+type_+"segmented\\" 
fpaths = glob.glob(datapath+"*.jpg")

endpath = "C:\\Users\\Camila\\Documents\\EDP\\Crop\\cropped_"+type_+"\\"

for i in range(len(fpaths)):#len(fpaths)):
    img = cv2.imread(fpaths[i],-1) # 0 = grayscale
    temp = fpaths[i]
    length = len(temp)
    pos = temp.rindex("\\")
    fname = temp[pos+1:length]
    pos = fname.rindex("_")
    fname = fname[0:pos]
    #aISIC_0009874_seg.jpg
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    height, width = img.shape
    
    #get positions
    y1 = int(height/5)*2
    y2 = math.ceil((height/5)*3)
    x1 = int(width/5)*2
    x2 = math.ceil(int(width/5)*3)
    maxy, miny = height, height
    maxx, minx = width, width
    
    max_val = []
    min_val = []
    #scan segment for highest and lowest values 
    for i in range(y1,y2): 
        line = img[i,:]
        line = line.tolist()
        if 0 in line:
            #temp_maxy = line.index(0)
            max_val.append(line.index(0))
    #        if temp_maxy < maxy:
    #            maxy = temp_maxy
        else:
            max_val.append(width)
        
        line = line[::-1]
        if 0 in line:
            #temp_miny = line.index(0)
            min_val.append(line.index(0))
    #        if temp_miny < miny:
    #            miny = temp_miny
        else:
            min_val.append(width)
    maxx = min(max_val)
    minx = width-min(min_val)
            
    max_val = []
    min_val = []
    for i in range(x1,x2): 
        line = img[:,i]
        line = line.tolist()
        if 0 in line:
            #temp_maxx = line.index(0)
            max_val.append(line.index(0))
    #        if temp_maxx < maxx:
    #            maxx = temp_maxx
        else:
            max_val.append(height)
        
        line = line[::-1]
        if 0 in line:
            #temp_minx = line.index(0)
            min_val.append(line.index(0))
    #        if temp_minx < minx:
    #            minx = temp_minx
        else:
            min_val.append(height)
    maxy = min(max_val)
    miny = height-min(min_val)
    
    temp_img = img.copy()
    for i in range(y1,y2):
        temp_img[i,:]=0
    for j in range(x1,x2):
        temp_img[:,j]=0
    
    if maxx==width:
        maxx=0
    if maxy==height:
        maxy=0
    
    if minx==0:
        minx=width
    if miny==0:
        miny=height
        
    img_uneven = img[maxy:miny,maxx:minx]
        
    side_len = int(max(miny-maxy,minx-maxx)/2)
    
    centre_y = miny-math.ceil((miny-maxy)/2)
    centre_x = minx-math.ceil((minx-maxx)/2)
    check_sides = numpy.array([centre_y-0,height-centre_y,centre_x-0,width-centre_x])
    
    if numpy.all(math.ceil(side_len*1.1) < check_sides):
        side_len = math.ceil(side_len*1.1)
    else:
        side_len = min(check_sides)
    #img[centre_y,centre_x]=
    imgc = img[(centre_y-side_len):(centre_y+side_len),(centre_x-side_len):(centre_x+side_len)]
    cv2.imshow('seg',temp_img)
    cv2.imshow('crop',imgc)
    cv2.imshow('img',img)
    cv2.imshow('img uneven',img_uneven)
    cv2.waitKey(0)
    cv2.imwrite(fname+"_c.jpg",imgc)



    
    


