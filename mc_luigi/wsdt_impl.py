from __future__ import division, print_function

import vigra
import fastfilters
import numpy as np

from scipy.ndimage.morphology import distance_transform_edt


# wrap vigra local maxima properly
def local_maxima(image, *args, **kwargs):
    assert image.ndim in (2, 3), "Unsupported dimensionality: {}".format(image.ndim)
    if image.ndim == 2:
        return vigra.analysis.localMaxima(image, *args, **kwargs)
    if image.ndim == 3:
        return vigra.analysis.localMaxima3D(image, *args, **kwargs)


# watershed on distance transform:
# seeds are generated on the inverted distance transform
# the probability map is used for growing
def compute_wsdt_segmentation(
    probability_map,
    threshold,
    sigma_seeds,
    min_segment_size=0,
    preserve_membrane=True
):

    # first, we compute the signed distance transform
    dt = signed_distance_transform(probability_map, threshold, preserve_membrane)

    # next, get the seeds via maxima on the (smoothed) distance transform
    seeds = seeds_from_distance_transform(dt, sigma_seeds)
    del dt  # remove the array name

    # run watershed on the pmaps with dt seeds
    max_label = iterative_inplace_watershed(probability_map, seeds, min_segment_size)
    return seeds, max_label


def signed_distance_transform(probability_map, threshold, preserve_membrane):

    # get the distance transform of the pmap
    binary_membranes = (probability_map >= threshold)
    distance_to_membrane = distance_transform_edt(np.logical_not(binary_membranes))

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
        distance_transform = fastfilters.gaussianSmoothing(distance_transform, sigma_seeds)

    # If any seeds end up on the membranes, we'll remove them.
    # This is more likely to happen when the distance transform was generated with preserve_membrane_pmaps=True
    membrane_mask = (distance_transform < 0)

    local_maxima(distance_transform, allowPlateaus=True, allowAtBorder=True, marker=np.nan, out=distance_transform)
    distance_transform = np.isnan(distance_transform)

    distance_transform[membrane_mask] = 0

    return vigra.analysis.labelMultiArrayWithBackground(distance_transform.view('uint8'))


def iterative_inplace_watershed(hmap, seeds, min_segment_size):

    _, max_label = vigra.analysis.watershedsNew(hmap, seeds=seeds, out=seeds)

    if min_segment_size:

        segments, counts = np.unique(seeds, return_counts=True)

        # mask segments which are smaller than min_segment size
        mask = np.ma.masked_array(seeds, np.in1d(seeds, segments[counts < min_segment_size])).mask
        seeds[mask] = 0

        _, max_label = vigra.analysis.watershedsNew(hmap, seeds=seeds, out=seeds)

        # Remove gaps in the list of label values.
        _, max_label, _ = vigra.analysis.relabelConsecutive(seeds,
                                                            start_label=0,
                                                            out=seeds,
                                                            keep_zeros=False)

    return max_label


if __name__ == '__main__':
    # from volumina_viewer import volumina_n_layer
    from wsdt import wsDtSegmentation
    from concurrent import futures
    import time

    pmap_path = '/home/consti/Work/data_neuro/cache/cremi/sample_A_train/inp1.h5'
    pmap = vigra.readHDF5(pmap_path, 'data')

    sigma = 1.6
    min_seg = 25
    th = 0.2

    def process_parallel(ws_fu):

        seg = np.zeros_like(pmap, dtype='uint32')

        def seg_z(z):
            seg_z, _ = ws_fu(pmap[z])
            seg[z] = seg_z

        with futures.ThreadPoolExecutor(max_workers=8) as tp:
            tasks = [tp.submit(seg_z, z) for z in range(seg.shape[0])]
            [t.result() for t in tasks]

        return seg

    from functools import partial
    ws0 = partial(
        compute_wsdt_segmentation,
        threshold=th,
        sigma_seeds=sigma,
        min_segment_size=min_seg
    )
    t0 = time.time()
    seg0 = process_parallel(ws0)
    print("T0:", time.time() - t0, 's')

    ws1 = partial(
        wsDtSegmentation,
        pmin=th,
        minMembraneSize=0,
        minSegmentSize=min_seg,
        sigmaMinima=sigma,
        sigmaWeights=0.,
        groupSeeds=False,
        preserve_membrane_pmaps=True,
        grow_on_pmap=True
    )
    t1 = time.time()
    seg1 = process_parallel(ws1)
    print("T1:", time.time() - t1, 's')

    assert seg0.shape == seg1.shape
    print(seg0.shape)

    # volumina_n_layer([pm, seg0])
