# -*- coding: utf-8 -*-
"""
CREDIT TO

@author:http://funcvis.org/blog/?p=51
"""
import cv2
import numpy as np
def createMatchedFilterBank(K, n = 12):
    '''
    Given a kernel, create matched filter bank
    '''
    rotate = 180 / n
    center = (K.shape[1] / 2, K.shape[0] / 2)
    cur_rot = 0
    kernels = [K]

    for i in range(1, n):
        cur_rot += rotate
        r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
        k = cv2.warpAffine(K, r_mat, (K.shape[1], K.shape[0]))
        kernels.append(k)
        #cur_rot += rotate
    return kernels

def applyFilters(im, kernels):
    '''
    Given a filter bank, apply them and record maximum response
    '''
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)