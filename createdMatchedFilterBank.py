# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:19:43 2018

@author: Camila
"""

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

    return kernels