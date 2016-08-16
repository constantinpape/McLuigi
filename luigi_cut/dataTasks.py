# Multicut Pipeline implemented with luigi
# Tasks for providing the input data

import luigi
from customTargets import HDF5VolumeTarget, RagTarget

from pipelineParameter import PipelineParameter
from toolsLuigi import config_logger

import logging

import numpy as np
import vigra
import os
import h5py
import time

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


class InputData(luigi.Task):
    """
    Task for loading external input data, e.g. raw data or probability maps.
    For HDF5 input.
    """

    path = luigi.Parameter()
    key  = luigi.Parameter(default = "data")
    # the dtype, should either be float32 or uint8
    dtype = luigi.Parameter(default = np.float32)

    def run(self):
        f = h5py.File(self.path, 'r')
        assert key in f.keys(), key + " , " + f.keys()
        dset = f[key]
        # TODO is this ok ?
        if self.dtype != f.dtype:
            print "InputData task, loading data from", self.path
            print "Changing dtype from", self.dtype, "to", f.dtype
            self.dtype = dset.dtype


    def output(self):
        """
        Returns the target output.

        :return: Target output
        :rtype: object( :py:class: HDF5Target)
        """

        return HDF5VolumeTarget(self.path, self.key, self.dtype )



class ExternalSegmentation(luigi.Task):
    """
    Task for loading external segmentation from HDF5.
    """

    # Path to the segmentation
    path = luigi.Parameter()
    key  = luigi.Parameter(default = "data")
    # the dtype, should either be uint32 or uint64
    dtype = luigi.Parameter(default = np.uint32)

    def run(self):

        f = h5py.File(self.path, 'r')
        assert key in f.keys(), key + " , " + f.keys()
        dset = f[key]
        # TODO is this ok ?
        if self.dtype != dset.dtype:
            print "ExternalSegmentation task, loading data from", self.path
            print "Changing dtype from", self.dtype, "to", dset.dtype
            self.dtype = dset.dtype


    def output(self):
        return HDF5VolumeTarget(self.path, self.key, self.dtype )



class ExternalSegmentationLabeled(luigi.Task):
    """
    Task for loading external segmentation from HDF5.
    Perform a label Volume and cache
    """

    # Path to the segmentation
    path = luigi.Parameter()
    key  = luigi.Parameter(default = "data")
    # the dtype, should either be uint32 or uint64
    dtype = luigi.Parameter(default = np.uint32)

    def requires(self):
        return ExternalSegmentation(path, key, dtype)

    def run(self):

        seg_array = self.input()
        # FIXME this is only feasible for small enough data, for larger data we need blockwise connected components
        # FIXME vigra mightt do wrong things for this axis order AND we can use the sliced structure also for more efficient relabeling!
        seg = vigra.analysis.labelVolume(seg_array.read([0,0,0],seg_array.shape)) - 1
        self.output().open()
        self.output().write( [0,0,0], seg)

    def output(self):
        save_path = os.path.join( PipelineParameter().cache,
                os.path.split(self.path)[1][:-3] + "_labeled.h5" )
        return HDF5VolumeTarget( save_path, self.key, self.dtype)



class DenseGroundtruth(luigi.Task):
    """
    Task for loading external groundtruth from HDF5.
    """

    path = luigi.Parameter()
    key  = luigi.Parameter(default = "data")
    # the dtype, should either be uint32 or uint64
    dtype = luigi.Parameter(default = np.uint32)

    def requires(self):
        return ExternalSegmentation(path, key, dtype)

    def run(self):

        gt_array = self.input()
        # FIXME this is only feasible for small enough data, for larger data we need blockwise connected components
        gt = vigra.analysis.labelVolumeWithBackground( gt_array.read([0,0,0],gt_array.shape)
        self.output.open()
        self.output().write( [0,0,0], gt)

    def output(self):
        save_path = os.path.join( PipelineParameter().cache,
                os.path.split(self.path)[1] )
        return HDF5VolumeTarget( save_path, key, dtype )


class StackedRegionAdjacencyGraph(luigi.Task):
    """
    Task for building the RAG
    """

    pathToSeg = luigi.Parameter()
    keyToSeg = luigi.Parameter(default = "data")

    # not really necessary right now, but maybe the rag syntax will change
    def requires(self):
        return ExternalSegmentation(self.pathToSeg)


    def run(self):

        # get the number of labels
        self.input().open()
        # we assume that the last slice has the highest
        shape = self.input().shape()
        seg_last = self.input().read( [shape[0]-1,0,0], shape )

        n_labels = seg_last.max() + 1

        t_rag = time.time()
        rag = nifty.graph.rag.gridRagStacked2DHdf5( self.input().get(), n_labels) # nThreads = -1, could also make this accessible
        t_rag = time.time() - t_rag

        workflow_logger.info("Computed RAG in " + str(t_rag) + " s")

        self.output().write( rag, pathToSeg, keyToSeg)



    def output(self):
        return StackedRagTarget( os.path.join(PipelineParameter().cache, "RegionAdjacencyGraph.h5") )
