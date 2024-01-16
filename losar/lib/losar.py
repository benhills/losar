#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 2024

@author: benhills
"""

import numpy as np

from scipy.ndimage import gaussian_filter as gf

from losar.lib.sar_functions import (doppler_centroid, layer_optimization,
                                     get_optimal_frequencies)


def losar(image, N,
          nus=np.linspace(-np.pi, np.pi, 100),
          dx=None,
          layer_finding='doppler',
          gaussian_filter=False,
          gf_window=[5, 5],
          verbose=False):
    """

    """

    snum, tnum = np.shape(image)

    losar_image = np.empty((2, snum, tnum))

    for tidx in np.arange(tnum):

        if verbose:
            print(tidx, end=' ')

        image_sub = image[:, tidx-N//2:tidx+N//2]

        if layer_finding == 'doppler':
            P, nus = doppler_centroid(image_sub, dx=dx)
        elif layer_finding == 'stack':
            P = layer_optimization(image_sub, nus)

        if gaussian_filter:
            P = gf(P, gf_window)

        p_best, f_best = get_optimal_frequencies(P, nus)

        losar_image[:, tidx, 0] = p_best
        losar_image[:, tidx, 1] = f_best

    return losar_image
