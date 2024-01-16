#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 2024

@author: benhills
"""

import numpy as np

"""
Supplemental functions for the losar processing library
"""

def dB(P):
    """
    Convert power to decibels

    Parameters
    ----------
    P:  float,  Power
    """
    return 10.*np.log10(P)
