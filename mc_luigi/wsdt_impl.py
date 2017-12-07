from __future__ import division, print_function

import vigra
# FIXME weird segfault that occurs when including nifty build with
# `WITH_FASTFILTERS` and importing fastfilters
# import fastfilters
import numpy as np

from skimage.feature import peak_local_max
from scipy.ndimage.morphology import distance_transform_edt


# wrap vigra local maxima properly
def local_maxima(array):
    assert array.ndim in (2, 3), "Unsupported dimensionality: {}".format(array.ndim)
    if array.ndim == 2:
        # this is considerably faster than the vigra implementation, but only works in 2d
        return peak_local_max(array, indices=False, exclude_border=False)
        #return vigra.analysis.localMaxima(array,
        #                                  allowAtBorder=True,
        #                                  allowPlateaus=True,
        #                                  marker=1).astype('bool')
    if array.ndim == 3:
        return vigra.analysis.localMaxima3D(array,
                                            allowAtBorder=True,
                                            allowPlateaus=True,
                                            marker=1).astype('bool')


# watershed on distance transform:
# seeds are generated on the inverted distance transform
# the probability map is used for growing
def compute_wsdt_segmentation(probability_map,
                              threshold,
                              sigma_seeds,
                              min_segment_size=0,
                              preserve_membrane=True,
                              start_label=0):

    # first, we compute the signed distance transform
    dt = signed_distance_transform(probability_map, threshold, preserve_membrane)

    # next, get the seeds via maxima on the (smoothed) distance transform
    seeds = seeds_from_distance_transform(dt, sigma_seeds)
    del dt  # remove the array name

    # run watershed on the pmaps with dt seeds
    max_label = iterative_inplace_watershed(probability_map, seeds, min_segment_size, start_label)
    return seeds, max_label


def signed_distance_transform(probability_map, threshold, preserve_membrane):

    # get the distance transform of the pmap
    binary_membranes = (probability_map >= threshold)
    # distance_to_membrane = distance_transform_edt(np.logical_not(binary_membranes)).astype('float32',copy=False)
    distance_to_membrane = vigra.filters.distanceTransform(binary_membranes.astype('uint32'))

    # Instead of computing a negative distance transform within the thresholded membrane areas,
    # Use the original probabilities (but inverted)
    if preserve_membrane:
        distance_to_membrane[binary_membranes] = -probability_map[binary_membranes]

    # Compute the negative distance transform and substract it from the distance transform
    else:
        distance_to_nonmembrane = vigra.filters.distanceTransform(binary_membranes)

        # Combine the inner/outer distance transforms
        distance_to_nonmembrane[distance_to_nonmembrane > 0] -= 1
        distance_to_membrane[:] -= distance_to_nonmembrane

    return distance_to_membrane


def seeds_from_distance_transform(distance_transform, sigma_seeds):

    # we are not using the dt after this point, so it's ok to smooth it
    # and later use it for calculating the seeds
    if sigma_seeds > 0.:
        # distance_transform = fastfilters.gaussianSmoothing(distance_transform, sigma_seeds)
        distance_transform = vigra.filters.gaussianSmoothing(distance_transform, sigma_seeds)

    # If any seeds end up on the membranes, we'll remove them.
    # This is more likely to happen when the distance transform
    # was generated with preserve_membrane_pmaps=True
    membrane_mask = (distance_transform < 0)
    distance_transform = local_maxima(distance_transform)
    distance_transform[membrane_mask] = False
    return vigra.analysis.labelMultiArrayWithBackground(distance_transform.view('uint8'))


def iterative_inplace_watershed(hmap, seeds, min_segment_size, start_label=0):

    _, max_label = vigra.analysis.watershedsNew(hmap, seeds=seeds, out=seeds)

    if min_segment_size:

        segments, counts = np.unique(seeds, return_counts=True)

        # mask segments which are smaller than min_segment size
        mask = np.ma.masked_array(seeds, np.in1d(seeds, segments[counts < min_segment_size])).mask
        seeds[mask] = 0

        _, max_label = vigra.analysis.watershedsNew(hmap, seeds=seeds, out=seeds)

        # Remove gaps in the list of label values.
        _, max_label, _ = vigra.analysis.relabelConsecutive(seeds,
                                                            start_label=start_label,
                                                            out=seeds,
                                                            keep_zeros=False)
    return max_label
