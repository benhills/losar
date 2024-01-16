#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 2024

@author: benhills
"""

import numpy as np

from scipy.signal import periodogram

from .supplemental import dB


def incoherent_average(image):
    """

    """
    return np.sum(abs(image)**2.,axis=1) / np.shape(image)[1]


def coherent_summation(image, phi=0.):
    """
    Sum the complex image across traces, coherently (account for phase)


    """
    snum,tnum = np.shape(image)

    if phi != 0.: # If phi is 0 there is no rotation so skip
        # phase offset for each trace
        phi_traces = np.linspace(tnum/2.*phi,-tnum/2.*phi,tnum)
        # get the amplitude and phase of the image
        R = abs(image)
        theta = np.angle(image)
        # add a phase offset based on trace distance from center of aperture
        theta += np.tile(phi_traces,(snum,1))
        # recalculate the complex image from amplitude and phase
        image = R*np.exp(1j*theta)

    return np.sum(image, axis=1)


def layer_optimization(image,phis):
    """

    """
    snum,tnum = np.shape(image)
    p_stack = np.empty((0,snum))
    for phi in phis:
        p_single = coherent_summation(image,phi=phi)
        p_stack = np.append(p_stack,[p_single],axis=0)
    return np.transpose(p_stack)


def doppler_centroid(image,dx):
    """
    Get the doppler centroid for an image (typically for one synthetic aperture, not the entire radar profile)

    Parameters
    -----------
    image   data image within a single synthetic aperture
    dx      spatial step between traces

    Output
    -----------
    p_dop   power
    freq    frequencies
    """
    snum,tnum = np.shape(image)
    p_dop = np.empty((0,tnum-1))
    for i in range(snum):
        freq, power = periodogram(image[i],fs=1./dx)
        idx = np.argsort(freq[1:])
        p_dop = np.append(p_dop,[power[1:][idx]],axis=0)
    return p_dop,freq[idx]


def get_optimal_frequencies(image,f_dop,threshold=-190.):
    """

    Parameters
    -----------
    image   data image within a single synthetic aperture

    Output
    -----------
    f   frequencies
    p   power
    """
    snum,tnum = np.shape(image)
    f = f_dop[np.argmax(image,axis=1)]
    p = np.array([np.max(image[i]) for i in range(snum)])
    f[dB(p)<threshold] = np.nan

    return p,f
