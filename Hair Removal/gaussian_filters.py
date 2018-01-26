# -*- coding: utf-8 -*-
"""
CREDIT TO

@author:http://funcvis.org/blog/?p=51
"""
from _filter_kernel_mf_fdog import _filter_kernel_mf_fdog

def fdog_filter_kernel(L, sigma, t = 3): #first div
    '''
    K = - (x/(sqrt(2 * pi) * sigma ^3)) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
    '''
    return _filter_kernel_mf_fdog(L, sigma, t, False)

def gaussian_matched_filter_kernel(L, sigma, t = 3): # second div
    '''
    K =  1/(sqrt(2 * pi) * sigma ) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
    '''
    return _filter_kernel_mf_fdog(L, sigma, t, True)