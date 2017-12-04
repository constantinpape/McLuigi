from __future__ import print_function, division
# Multicut Pipeline implemented with luigi
# Tasks for providing the input data

import luigi

from .customTargets import VolumeTarget, StackedRagTarget
from .pipelineParameter import PipelineParameter
from .tools import config_logger, run_decorator

import logging

import numpy as np
import vigra
import os
import time

from concurrent import futures

from .wsdt_impl import compute_wsdt_segmentation  # , compute_wsdt_segmentation_with_mask

# import the proper nifty version
try:
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        import nifty_with_gurobi.graph.rag as nrag

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


###########################
# Tasks for data generation
###########################


# FIXME creates wrong offsets atm !!
# not ingestible by stacked rag
class WsdtSegmentation(luigi.Task):
    """
    Task for generating segmentation via wsdt.
    """

    pathToProbabilities = luigi.Parameter()
    keyToProbabilities = luigi.Parameter()
    dtype = luigi.Parameter(default='uint32')
    pathToMask = luigi.Parameter(default=None)
    keyToMask = luigi.Parameter(default='data')

    def requires(self):
        """
        Dependencies:
        """
        deps = {'pmap': InputData(self.pathToProbabilities)}
        if self.pathToMask is not None:
            deps['mask'] = InputData(self.pathToMask)
        return deps

    # TODO enable 3d wsdt for isotropic ppl
    @run_decorator
    def run(self):
        pmap = self.input()['pmap']
        pmap.open(self.keyToProbabilities)
        shape = pmap.shape(self.keyToProbabilities)
        out = self.output()
        out.open(shape=shape,
                 chunks=pmap.chunks(self.keyToProbabilities),
                 dtype=self.dtype)

        if self.pathToMask is None:
            self._run_wsdt2d_standard(pmap, out, shape)
        else:
            mask = self.input()['mask']
            mask.open(self.keyToMask)
            self._run_wsdt2d_with_mask(pmap, mask, out, shape)
            mask.close()

        out.close()
        pmap.close()

    def _run_wsdt2d_with_mask(self, pmap, mask, out, shape):
        # read the wsdt settings from ppl params
        ppl_params = PipelineParameter()
        threshold  = ppl_params.wsdtThreshold
        min_seg    = ppl_params.wsdtMinSeg
        sig_seeds  = ppl_params.wsdtSigSeeds
        invert     = ppl_params.wsdtInvert
        workflow_logger.info(
            "WsdtSegmentation: Running 2d dt watershed with mask with threshold %f" % threshold
        )

        def segment_slice(z):
            print("Slice", z, "/", shape[0])
            sliceStart = [z, 0, 0]
            sliceStop  = [z + 1, shape[1], shape[2]]
            pmap_z = pmap.read(sliceStart, sliceStop, self.keyToProbabilities).squeeze()
            mask_z = mask.read(sliceStart, sliceStop, self.keyToMask).squeeze().astype('bool')
            if invert:
                pmap_z = 1. - pmap_z
            seg, max_z = compute_wsdt_segmentation(pmap_z,
                                                   threshold,
                                                   sig_seeds,
                                                   min_seg)
            # alternative: use default watershed and mask after that
            seg[mask_z] = 0
            seg = vigra.analysis.labelMultiArrayWithBackground(seg)
            max_z = seg.max()
            out.write(sliceStart, seg[None, :, :])
            return max_z + 1

        n_workers = PipelineParameter().nThreads

        t_wsdt = time.time()
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(segment_slice, z) for z in range(shape[0])]
            offsets = [future.result() for future in tasks]
        workflow_logger.info("WsdtSegmentation: Running watershed took: %f s"
                             % (time.time() - t_wsdt))

        # accumulate the offsets for each slice
        offsets = np.roll(offsets, 1)
        offsets[0] = 0
        offsets = np.cumsum(offsets)

        if offsets[-1] >= 4294967295 and self.dtype == np.dtype('uint32'):
            print("WARNING: UINT32 overflow!")
            # we don't add the offset, but only save the non-offset file
            return False

        def add_offset(z, offset):
            sliceStart = [z, 0, 0]
            sliceStop  = [z + 1, shape[1], shape[2]]
            seg = out.read(sliceStart, sliceStop)
            mask_z = mask.read(sliceStart, sliceStop, self.keyToMask).astype('bool')
            seg[np.logical_not(mask_z)] += offset
            out.write(sliceStart, seg)

        t_off = time.time()
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(add_offset, z, offset) for z, offset in enumerate(offsets)]
            [t.result() for t in tasks]
        workflow_logger.info("WsdtSegmentation: Adding offsets took %f s" % (time.time() - t_off))
        return True

    def _run_wsdt2d_standard(self, pmap, out, shape):

        # read the wsdt settings from ppl params
        ppl_params = PipelineParameter()
        threshold   = ppl_params.wsdtThreshold
        min_seg     = ppl_params.wsdtMinSeg
        sig_seeds   = ppl_params.wsdtSigSeeds
        invert      = ppl_params.wsdtInvert
        workflow_logger.info(
            "WsdtSegmentation: Running standard 2d dt watershed with threshold %f" % threshold
        )

        def segment_slice(z):
            print("Slice", z, "/", shape[0])
            sliceStart = [z, 0, 0]
            sliceStop  = [z + 1, shape[1], shape[2]]
            pmap_z = pmap.read(sliceStart, sliceStop, self.keyToProbabilities).squeeze()
            if invert:
                pmap_z = 1. - pmap_z
            seg, max_z = compute_wsdt_segmentation(pmap_z, threshold, sig_seeds, min_seg)
            out.write(sliceStart, seg[None, :, :])
            return max_z + 1

        n_workers = PipelineParameter().nThreads

        t_wsdt = time.time()
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(segment_slice, z) for z in range(shape[0])]
            offsets = [future.result() for future in tasks]
        workflow_logger.info("WsdtSegmentation: Running watershed took: %f s"
                             % (time.time() - t_wsdt))

        # accumulate the offsets for each slice
        offsets = np.roll(offsets, 1)
        offsets[0] = 0
        offsets = np.cumsum(offsets)

        if offsets[-1] >= 4294967295 and self.dtype == np.dtype('uint32'):
            print("WARNING: UINT32 overflow!")
            # we don't add the offset, but only save the non-offset file
            return False

        def add_offset(z, offset):
            sliceStart = [z, 0, 0]
            sliceStop  = [z + 1, shape[1], shape[2]]
            seg = out.read(sliceStart, sliceStop)
            seg += offset
            out.write(sliceStart, seg)

        t_off = time.time()
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(add_offset, z, offsets[z]) for z in range(shape[0])]
            [t.result() for t in tasks]
        workflow_logger.info("WsdtSegmentation: Adding offsets took %f s" % (time.time() - t_off))
        return True

    def output(self):
        save_path = os.path.join(PipelineParameter().cache,
                                 "WsdtSegmentation_%s" %
                                 os.path.split(self.pathToProbabilities)[1][:-3])
        return VolumeTarget(save_path)


###########################
# Tasks for providing data
###########################


class InputData(luigi.Task):
    """
    Task for loading external input data, e.g. raw data or probability maps.
    """
    path = luigi.Parameter()

    def output(self):
        """
        Returns the target output.

        :return: Target output
        :rtype: object( :py:class: VolumeTarget)
        """
        return VolumeTarget(self.path)


class ExternalSegmentation(luigi.Task):
    """
    Task for loading external segmentation from HDF5.
    """

    # Path to the segmentation
    path = luigi.Parameter()

    # if the path does not exist, run the watershed on distance transform segmentation
    def requires(self):
        if not os.path.exists(self.path):

            # find the corresponding affinity maps from the inputs
            aff_prefix = os.path.split(self.path)[1].split('_')[1:]
            aff_prefix = '_'.join(aff_prefix)
            inputs = PipelineParameter().inputs['data']
            found_aff = False
            for in_path in inputs:
                if os.path.split(in_path)[1] == aff_prefix:
                    aff_path = in_path
                    found_aff = True
                    break
            if not found_aff:
                raise RuntimeError("Couldn't find affinty path for requested wsdt segmentation: %s" % self.path)
            return WsdtSegmentation(aff_path)

    def output(self):
        """
        Returns the target output.

        :return: Target output
        :rtype: object( :py:class: VolumeTarget)
        """
        return VolumeTarget(self.path)


class ExternalSegmentationLabeled(luigi.Task):
    """
    Task for loading external segmentation from HDF5.
    Perform a label Volume and cache.
    """

    # Path to the segmentation
    pathToSeg = luigi.Parameter()
    key  = luigi.Parameter(default="data")
    # the dtype, should either be uint32 or uint64
    dtype = luigi.Parameter(default="uint32")

    def requires(self):
        return ExternalSegmentation(self.pathToSeg)

    @run_decorator
    def run(self):

        seg_in = self.input()
        seg_in.open(self.key)

        shape = seg_in.shape()

        seg_out = self.output()
        seg_out.open(self.key, shape=shape, chunks=seg_in.chunks, dtype=self.dtype)

        def label_slice(z):
            begin = [z, 0, 0]
            end   = [z + 1, shape[1], shape[2]]
            seg_slice = seg_in.read(begin, end).squeeze()
            seg_slice = vigra.analysis.labelImage(seg_slice) - 1
            offset  = seg_slice.max()
            seg_out.write(begin, seg_slice[None, :, :])
            return offset

        n_workers = min(shape[0], PipelineParameter().nThreads)
        # n_workers = 1
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(label_slice, z) for z in range(shape[0])]

        # calculate the offsets for every slice
        offsets = np.array([task.result() for task in tasks], dtype='uint32')
        offsets = np.cumsum(offsets)
        # need to shift by 1 to the left and insert a 0
        offsets = np.roll(offsets, 1)
        offsets[0] = 0

        def add_offset(offset, z):
            begin = [z, 0, 0]
            end   = [z + 1, shape[1], shape[2]]
            seg_slice = seg_out.read(begin, end)
            seg_slice += offset
            seg_out.write(begin, seg_slice)
            return True

        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(add_offset, offsets[z], z) for z in range(shape[0])]
        [task.result() for task in tasks]

    def output(self):
        save_path = os.path.join(PipelineParameter().cache,
                                 os.path.split(self.pathToSeg)[1][:-3] + "_labeled.h5")
        return VolumeTarget(save_path)


class DenseGroundtruth(luigi.Task):
    """
    Task for loading external groundtruth from HDF5.
    """

    path = luigi.Parameter()
    key  = luigi.Parameter(default="data")
    # the dtype, should either be uint32 or uint64
    dtype = luigi.Parameter(default="uint32")

    def requires(self):
        return ExternalSegmentation(self.path)

    @run_decorator
    def run(self):

        gt = self.input()
        gt.open(self.key)

        # FIXME this will be problematic for ignore labels that are not 0
        # easiest fix is to reimplement 'relabelConsecutive' based on 'replace_from_dict'
        # and allow to keep arbitrary values fixed
        gt_labeled, _, _ = vigra.analysis.relabelConsecutive(
            gt.read([0, 0, 0], gt.shape()).astype('uint32', copy=False),
            start_label=1,
            keep_zeros=True
        )

        out = self.output()
        out.open(self.key, shape=gt.shape(), chunks=gt.chunks, dtype=self.dtype)
        out.write([0, 0, 0], gt_labeled)

        gt.close()
        out.close()

    def output(self):
        save_path = os.path.join(PipelineParameter().cache,
                                 os.path.split(self.path)[1][:-3] + "_labeled.h5")
        return VolumeTarget(save_path)


class StackedRegionAdjacencyGraph(luigi.Task):
    """
    Task for building the RAG
    """

    pathToSeg = luigi.Parameter()
    keyToSeg = luigi.Parameter(default="data")

    # not really necessary right now, but maybe the rag syntax will change
    def requires(self):
        return ExternalSegmentation(self.pathToSeg)

    @run_decorator
    def run(self):

        # get the number of labels
        seg = self.input()

        seg.open()
        shape = seg.shape()

        seg_last = seg.read([shape[0] - 1, 0, 0], shape)
        n_labels = seg_last.max() + 1

        t_rag = time.time()
        # nThreads = -1, could also make this accessible
        rag = nrag.gridRagStacked2DHdf5(seg.get(), n_labels, numberOfThreads=-1)
        t_rag = time.time() - t_rag

        self.output().write(rag, self.pathToSeg, self.keyToSeg)

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = "StackedRegionAdjacencyGraph_%s.h5" % segFile
        return StackedRagTarget(os.path.join(PipelineParameter().cache, save_path))
