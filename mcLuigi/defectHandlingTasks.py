# Multicut Pipeline implemented with luigi
# Taksks for defect and handling
import luigi

from customTargets import HDF5DataTarget, HDF5VolumeTarget
from dataTasks import InputData, ExternalSegmentation
from defectDetectionTasks import DefectSliceDetection

from pipelineParameter import PipelineParameter
from tools import config_logger

import logging
import json

import os
import time
import numpy as np
import vigra
import nifty

from concurrent import futures


# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


# map defect patches to overlapping superpixels
class DefectsToNodes(luigi.Task):

    pathToSeg = luigi.Parameter()

    def requires(self):
        return {'seg' : ExternalSegmentation(self.pathToSeg),
                'defects' : DefectSliceDetection(self.pathToSeg)} # need to tweek binThreshold w/ the default param

    def run(self):
        inp = self.input()
        seg = inp['seg']
        seg.open()
        defects = inp['defects']
        defects.open()

        assert seg.shape == defects.shape

        ny = long(seg.shape[1])
        nx = long(seg.shape[2])

        def defects_to_nodes_z(z):
            slice_begin = [long(z),0L,0L]
            slice_end   = [long(z+1),ny,nx]
            defect_mask = defects.read(slice_begin,slice_end)
            if 1 in defect_mask:
                seg_z = seg.read(slice_begin,slice_end)
                where_defect = defect_mask == 1
                return list(np.unique(seg_z[where_defect]))
            else:
                return []

        with futures.ThreadPoolExecutor(max_workers = PipelineParameter().nThreads) as executor:
            tasks = []
            for z in xrange(seg.shape[0]):
                tasks.append(executor.submit(defects_to_nodes_z,z))
            defect_nodes = []
            for fut in tasks:
                defect_nodes.extend(fut.result())

        self.output().write(np.array(defect_nodes,dtype='uint32'),'data')

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "DefectsToNodes_%s.h5" % (segFile,) )
        return HDF5DataTarget(save_path)


# modify the rag to isolate defected superpixels
# introduce skip edges over the defected superpixels in z
# TODO it is probably simpler to modify the plain graph that is fed to the mc instead of the rag
class ModifiedAdjacency(luigi.Task):

    self.pathToSeg = luigi.Parameter()

    def requires(self):
        return {'seg' : ExternalSegmentation(self.pathToSeg),
                'defect_nodes' : DefectsToNodes(self.pathToSeg)}

    def run(self):
        pass
        # TODO compute and save the skip edges as tuples (n1,n2) as well
        # as the resulting modified adjacency (as plain graph)

    def output(self):
        pass


# TODO implement
# calculate and insert features for the skip edges
class ModifiedFeatures(luigi.Task):

    def requires(self):
        pass


# TODO implement
# modify the mc problem by changing the graph to the 'ModifiedAdjacency'
# and setting xy-edges that connect defected with non-defected nodes to be maximal repulsive
class ModifiedMcProblem(luigi.Task):

    def requires(self):
        pass


# TODO implement
# replace the segmentation in completely defected slices with adjacent slice
class PostprocessDefectedSlices(luigi.Task):

    def requires(self):
        pass
