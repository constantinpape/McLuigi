# Multicut Pipeline implemented with luigi
# Tasks for providing the input data

import luigi
from CustomTargets import HDF5Target, RagTarget

from PipelineParameter import *

import logging

import numpy as np
import vigra
import os
import h5py

# init the workflow logger
from customLogging import config_logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

# we do nasty stuff and change the key inside the dataset to data
# don't know, if this is wise, but this is the most lazy option...
class InputData(luigi.Task):
    """
    Task for loading external input data, e.g. raw data or probability maps.
    For HDF5 input.
    """

    PathToData = luigi.Parameter()

    def run(self):
        f = h5py.File(self.PathToData, 'r+')
        keys = f.keys()
        f.close()

        if len(keys) > 1:
            raise RuntimeError("Can only handle single key in input data.")
        key = keys[0]

        if key != "data":
            f["data"] = f[key]
            del f[key]

    def output(self):
        """
        Returns the target output.

        :return: Target output
        :rtype: object( :py:class: HDF5Target)
        """

        return HDF5Target(self.PathToData)



class ExternalSegmentation(luigi.Task):
    """
    Task for loading external segmentation from HDF5.
    """

    # Path to the segmentation
    PathToSeg = luigi.Parameter()

    def run(self):
        f = h5py.File(self.PathToData, 'r+')
        keys = f.keys()
        f.close()

        if len(keys) > 1:
            raise RuntimeError("Can only handle single key in input data.")
        key = keys[0]

        if key != "data":
            f["data"] = f[key]
            del f[key]


    def output(self):
        return HDF5Target( self.PathToSeg  )


class ExternalSegmentationLabeled(luigi.Task):
    """
    Task for loading external segmentation from HDF5.
    Perform a label Volume and cache
    """

    # Path to the segmentation
    PathToSeg = luigi.Parameter()

    def run(self):

        f = h5py.File(self.PathToSeg, 'r')
        keys = f.keys()
        f.close()
        if len(keys) > 1:
            raise RuntimeError("Can only handle single key in input data.")
        key = keys[0]

        seg = vigra.readHDF5(self.PathToSeg, key).astype(np.uint32)
        seg = vigra.analysis.labelVolume(seg) - 1
        self.output().write(seg)

    def output(self):
        save_path = os.path.join( PipelineParameter().cache,
                os.path.split(self.PathToSeg)[1] )
        return HDF5Target( save_path  )



class DenseGroundtruth(luigi.Task):
    """
    Task for loading external groundtruth from HDF5.
    """

    PathToGt = luigi.Parameter()

    def run(self):

        f = h5py.File(PathToSeg, 'r')
        keys = f.keys()
        f.close()
        if len(keys) > 1:
            raise RuntimeError("Can only handle single key in input data.")
        key = keys[0]

        gt = vigra.readHDF5(self.PathToSeg, key).astype(np.uint32)
        gt = vigra.analysis.labelVolumeWithBackground(gt)
        self.output().write(gt)

    def output(self):
        save_path = os.path.join( PipelineParameter().cache,
                os.path.split(self.PathToSeg)[1] )
        return HDF5Target( save_path  )


class RegionAdjacencyGraph(luigi.Task):
    """
    Task for building the RAG
    """

    PathToSeg = luigi.Parameter()

    def requires(self):
        return ExternalSegmentationLabeled(self.PathToSeg)


    def run(self):

        t_rag = time.time()

        seg = self.input().read()
        self.output().write(vigra.graphs.regionAdjacencyGraph(
            vigra.graphs.gridGraph(seg.shape[0:3]), seg))

        t_rag = time.time() - t_rag
        workflow_logger("Computed RAG in " + str(t_rag) + " s")


    def output(self):
        return RagTarget( os.path.join(PipelineParameter().cache,
            os.path.split(self.PathToSeg)[1] + "_rag.h5") )
