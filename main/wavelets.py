#!/usr/bin/nv python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: wavelets.py
#  Purpose: Signal Analysis
#   Author: Vidal Gonzalez P
#    Email: vidalgonz8@gmail.com
# --------------------------------------------------------------------
"""

        Some useful functions.

"""
import numpy as np

def ormsby(freqs, tvec):
    '''
    Computes ormsby amplitude wavelet.
    '''
    f1, f2, f3, f4 = freqs
    a1 = ((np.pi * f4)**2 / (np.pi * f4 - np.pi * f3)) * np.sinc(f4 * tvec)**2
    a2 = ((np.pi * f3)**2 / (np.pi * f4 - np.pi * f3)) * np.sinc(f3 * tvec)**2
    b1 = ((np.pi * f2)**2 / (np.pi * f2 - np.pi * f1)) * np.sinc(f2 * tvec)**2
    b2 = ((np.pi * f1)**2 / (np.pi * f2 - np.pi * f1)) * np.sinc(f1 * tvec)**2
    return((a1 - a2) - (b1 - b2))

def ricker(fc, tvec):
    '''
    Computes ricker amplitude wavelet
    '''
    return ((1.0 - 2.0 * np.pi**2 * fc**2 * tvec**2) * \
        np.exp(-np.pi**2 * fc**2 * tvec**2))

def main():
    pass

if (__name__ == '__main__'):
    main()


