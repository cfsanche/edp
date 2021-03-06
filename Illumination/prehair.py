# -*- coding: utf-8 -*-
"""
NOTES:
    - must put "\\" to open file paths 
    
    @author: Camila and Hadeer
"""
import numpy as np
import cv2 #step1
#STEP 1: Homomorphic Filter 
#import image in colour 
#transform image into CIE L*a*b* colour space
#appy homomorphic filter to the L* (luminance) plane
# REFERENCE: https://blogs.mathworks.com/steve/2013/06/25/homomorphic-filtering-part-1/
# https://www.youtube.com/watch?v=Q2_PD6MGOoM
# http://biomedpharmajournal.org/vol7no2/image-sharpening-by-gaussian-and-butterworth-high-pass-filter/
# http://www.ipcsit.com/vol45/015-ICIKM2012-M0029.pdf

img = cv2.imread('C:\\Users\\Camila\\Documents\\EDP\\Dataset\\ISIC_0000028.jpg',1) # 0 = grayscale
# ISIC_0009953
# ISIC_0000042
# ISIC_0000043

# ISIC_0000027
# ISIC_0000028
# ISIC_0000029

img1 = np.float32(img)
img1 = img1/255
rows,cols,dim=img1.shape

rh, rl, cutoff = 1.5,0.5,10

imgYCrCb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
y,cr,cb = cv2.split(imgYCrCb)

y_log = np.log(y+0.01)

y_fft = np.fft.fft2(y_log) #gaussian filter is centered so centre this

y_fft_shift = np.fft.fftshift(y_fft)


DX = 0.25 # cutoff/255
G = np.ones((rows,cols))
for i in range(rows):
    for j in range(cols):
        G[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*DX**2))))+rl

result_filter = G * y_fft_shift

result_interm = np.float32(np.exp(np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))))

result = cv2.merge((result_interm,cr,cb))
result = cv2.cvtColor(result,cv2.COLOR_YCrCb2BGR)
img = cv2.resize(img,(752,565))
result_save = result*255
cv2.imwrite('correction_test.jpg',result_save)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#result = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
result_show = cv2.resize(result,(np.around(cols/2).astype(int),np.around(rows/2).astype(int)))
cv2.imshow('image1',img)
cv2.imshow('image2',result_show)
cv2.waitKey(0) #necessary? -yes
cv2.destroyAllWindows()
result_save = result*255
cv2.imwrite('correction_test5.jpg',result_save)
