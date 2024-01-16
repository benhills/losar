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
    Incoherent stack (amplitude only, no phase tracking)

    Parameters
    -----------
    image           2d array (snum,tnum); data image to stack

    Output
    -----------
    stacked_image   2d array (snum,tnum)
    """

    return np.sum(abs(image)**2., axis=1) / np.shape(image)[1]


def coherent_summation(image, phi=0.):
    """
    Sum the complex image across traces, coherently (account for phase)

    Parameters
    -----------
    image           2d array (snum,tnum); data image to stack
    phi             float; phase shift between traces

    Output
    -----------
    stacked_image   2d array (snum,tnum)
    """

    snum, tnum = np.shape(image)

    if phi != 0.:  # If phi is 0 there is no rotation so skip
        # phase offset for each trace
        phi_traces = np.linspace(tnum/2.*phi, -tnum/2.*phi, tnum)
        # get the amplitude and phase of the image
        R = abs(image)
        theta = np.angle(image)
        # add a phase offset based on trace distance from center of aperture
        theta += np.tile(phi_traces, (snum, 1))
        # recalculate the complex image from amplitude and phase
        image = R*np.exp(1j*theta)

    return np.sum(image, axis=1)


def losar_stack(image, phis):
    """
    Option 1 of 2 for the core of the LoSAR algorithm
    Coherent sum, stacking across a range of viable dips
    Castelletti et al. (2019)

    Parameters
    -----------
    image       2d array (snum,tnum); data image to stack
    phis        1d array; phase offsets

    Output
    -----------
    p_stack     2d array; stacked power for all depths and at all phis
    """

    snum, tnum = np.shape(image)
    P_stack = np.empty((0, snum))
    for phi in phis:  # TODO: convert to wavenumbers
        P_single = coherent_summation(image, phi=phi)
        P_stack = np.append(P_stack, [P_single], axis=0)
    return np.transpose(P_stack)


def losar_doppler(image, dx):
    """
    Option 2 of 2 for the core of the LoSAR algorithm
    Get the periodograms relative to doppler centroid for an image
    MacGregor et al. (2015) (Radiostratigraphy of Greenland)

    Parameters
    -----------
    image       2d array (snum,tnum); data image to stack
    dx          spatial step between traces

    Output
    -----------
    P_dop       2d array; power from periodogram
    nus         1d array; sorted wavenumbers
    """
    snum, tnum = np.shape(image)
    P_dop = np.empty((0, tnum-1))
    for i in range(snum):
        nus, power = periodogram(image[i], fs=1./dx)
        idx = np.argsort(nus[1:])
        P_dop = np.append(P_dop, [power[1:][idx]], axis=0)
    return P_dop, nus[idx]


def get_optimal_wavenumbers(P, nus_full, threshold=-190.):
    """
    Get the 'best' wavenumbers based on the most stacked power
    from one of the functions above.

    Parameters
    -----------
    P           2d array; Stacked power (from one of the losar functions above)
    nus_full    1d array; wavenumbers
    threshold   float; cutoff value for stacked power
                        (low power assign slope as nan)

    Output
    -----------
    p_best      power
    nu_best   frequencies
    """
    snum, N = np.shape(P)
    nu_best = nus_full[np.argmax(P, axis=1)]
    p_best = np.array([np.max(P[i]) for i in range(snum)])
    nu_best[dB(p_best) < threshold] = np.nan

    return p_best, nu_best
