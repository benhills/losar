#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 2024

@author: benhills
"""

import numpy as np

from scipy.ndimage import gaussian_filter as gf

from losar.lib.sar_functions import (losar_doppler, losar_stack,
                                     get_optimal_wavenumbers)


def losar(image, N,
          nus=np.linspace(-np.pi, np.pi, 100),
          dx=None,
          layer_finding='doppler',
          gaussian_filter=False,
          gf_window=[5, 5],
          verbose=False):
    """
    The full LoSAR algorithm for an ice-penetrating radar profile.
    Based on Castelletti et al. (2019)
    The core of the processing functions are in a separate script.

    Parameters
    -----------
    image           2d array; input image
    N               int; SAR aperture (number of traces)
    nus             1d array; wavenumbers
    dx              float; trace separation (distance)
    layer_finding   string; choice of slope finding algorithm
    gaussian_filter bool; decide to smooth the stacked power/wavenumber image
    gf_window       list; smoothing window
    verbose         bool; print output

    Output
    -----------
    losar_image     3d array; 2 images the size of the input image
                                first is the stacked power
                                second is the extracted slopes
    """

    # shape of input image, num_samples, num_traces
    snum, tnum = np.shape(image)

    # pre-filled output array
    losar_image = np.empty((2, snum, tnum))

    # Loop through all traces
    for tidx in np.arange(tnum):
        if verbose:  # print trace number; show the loop is going
            print(tidx, end=' ')
        # subset of the image with the aperture length around the given trace
        image_sub = image[:, max(0, tidx-N//2): min(tnum, tidx+N//2)]

        # Get stacked power using one of the imported losar functions
        if layer_finding == 'doppler':
            P, nus = losar_doppler(image_sub, dx=dx)
        elif layer_finding == 'stack':
            P = losar_stack(image_sub, nus)

        # Smooth the stacked power image
        if gaussian_filter:
            P = gf(P, gf_window)

        # Get the 'best' layer dip and power stacked along that dip
        p_best, f_best = get_optimal_wavenumbers(P, nus)

        # Save the stacked power and layer dip to output array
        losar_image[0, :, tidx] = p_best
        losar_image[1, :, tidx] = f_best

    return losar_image
