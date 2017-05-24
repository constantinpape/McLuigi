# Multicut Pipeline implemented with luigi
# Tasks for providing the input data

import luigi
from customTargets import HDF5VolumeTarget, StackedRagTarget

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging

import numpy as np
import vigra
import os
import h5py
import time

from concurrent import futures

# import the proper nifty version
try:
    import nifty
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex as nifty
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        import nifty_with_gurobi as nifty
        import nifty_with_gurobi.graph.rag as nrag

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


class InputData(luigi.Task):
    """
    Task for loading external input data, e.g. raw data or probability maps.
    For HDF5 input.
    """

    path = luigi.Parameter()
    key  = luigi.Parameter(default="data")
    # the dtype, should either be float32 or uint8
    dtype = luigi.Parameter(default="float32")

    def run(self):
        pass

    def output(self):
        """
        Returns the target output.

        :return: Target output
        :rtype: object( :py:class: HDF5Target)
        """

        with h5py.File(self.path, 'r') as f:
            assert self.key in f.keys(), self.key + " , " + str(f.keys())
            dset = f[self.key]

            if np.dtype(self.dtype) != np.dtype(dset.dtype):
                workflow_logger.debug("InputData: task, loading data from %s" % self.path)
                workflow_logger.debug("InputData: changing dtype from %s to %s" % (self.dtype, dset.dtype))
                self.dtype = dset.dtype

        return HDF5VolumeTarget(self.path, self.dtype, self.key)


class ExternalSegmentation(luigi.Task):
    """
    Task for loading external segmentation from HDF5.
    """

    # Path to the segmentation
    path = luigi.Parameter()
    key  = luigi.Parameter(default="data")
    # the dtype, should either be uint32 or uint64
    dtype = luigi.Parameter(default="uint32")

    def run(self):
        pass

    def output(self):
        """
        Returns the target output.

        :return: Target output
        :rtype: object( :py:class: HDF5Target)
        """

        assert os.path.exists(self.path), self.path
        with h5py.File(self.path, 'r') as f:
            assert self.key in f.keys(), self.key + " , " + str(f.keys())
            dset = f[self.key]

            if np.dtype(self.dtype) != np.dtype(dset.dtype):
                workflow_logger.debug("ExternalSegmentation: loading data from %s" % self.path)
                workflow_logger.debug("ExternalSegmentation: changing dtype from %s to %s" % (self.dtype, dset.dtype))
                self.dtype = dset.dtype

        return HDF5VolumeTarget(self.path, self.dtype, self.key, compression=PipelineParameter().compressionLevel)


class ExternalSegmentationLabeled(luigi.Task):
    """
    Task for loading external segmentation from HDF5.
    Perform a label Volume and cache.
    """

    # Path to the segmentation
    pathToSeg = luigi.Parameter()
    keyToSeg  = luigi.Parameter(default="data")
    # the dtype, should either be uint32 or uint64
    dtype = luigi.Parameter(default="uint32")

    def requires(self):
        return ExternalSegmentation(self.pathToSeg, self.keyToSeg, self.dtype)

    @run_decorator
    def run(self):

        seg_in = self.input()
        seg_in.open()

        shape = seg_in.shape

        seg_out = self.output()
        seg_out.open(shape)

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
            tasks = [executor.submit(label_slice, z) for z in xrange(shape[0])]

        # calculate the offsets for every slice
        offsets = np.array([task.result() for task in tasks], dtype='uint32')
        offsets = np.cumsum(offsets)
        # need to shift by 1 to the left and insert a 0
        offsets = np.roll(offsets, 1)
        offsets[0] = 0

        def addOffset(offset, z):
            begin = [z, 0, 0]
            end   = [z + 1, shape[1], shape[2]]
            seg_slice = seg_out.read(begin, end)
            seg_slice += offset
            seg_out.write(begin, seg_slice)
            return True

        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(addOffset, offsets[z], z) for z in xrange(shape[0])]
        [task.result() for task in tasks]

    def output(self):
        f = h5py.File(self.path, 'r')
        assert self.key in f.keys(), self.key + " , " + f.keys()
        dset = f[self.key]

        if np.dtype(self.dtype) != np.dtype(dset.dtype):
            workflow_logger.debug("ExternalSegmentationLabeled: loading data from %s" % self.path)
            workflow_logger.debug(
                "ExternalSegmentationLabeled: changing dtype from %s to %s" % (self.dtype, dset.dtype)
            )
            self.dtype = dset.dtype

        save_path = os.path.join(
            PipelineParameter().cache,
            os.path.split(self.pathToSeg)[1][:-3] + "_labeled.h5"
        )
        return HDF5VolumeTarget(save_path, self.dtype, compression=PipelineParameter().compressionLevel)


class DenseGroundtruth(luigi.Task):
    """
    Task for loading external groundtruth from HDF5.
    """

    path = luigi.Parameter()
    key  = luigi.Parameter(default="data")
    # the dtype, should either be uint32 or uint64
    dtype = luigi.Parameter(default="uint32")

    def requires(self):
        return ExternalSegmentation(self.path, self.key, self.dtype)

    @run_decorator
    def run(self):

        gt = self.input()
        gt.open()
        # label volume causes problems for cremi...
        gt_labeled, _, _ = vigra.analysis.relabelConsecutive(
            gt.read([0, 0, 0], gt.shape()),
            start_label=0,
            keep_zeros=False
        )

        out = self.output()
        out.open(gt.shape())
        out.write([0, 0, 0], gt_labeled)
        out.close()

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            os.path.split(self.path)[1][:-3] + "_labeled.h5"
        )
        return HDF5VolumeTarget(save_path, self.dtype, compression=PipelineParameter().compressionLevel)


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
