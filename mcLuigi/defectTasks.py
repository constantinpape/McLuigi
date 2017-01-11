# Multicut Pipeline implemented with luigi
# Taksks for defect detection and handling
import luigi

from customTargets import HDF5DataTarget, HDF5VolumeTarget
from dataTasks import InputData, ExternalSegmentation
#from miscTasks import EdgeIndications

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


class OversegmentationStatistics(luigi.Task):

    pathToSeg = luigi.Parameter()
    patchSize = luigi.Parameter()

    def requires(self):
        return ExternalSegmentation(self.pathToSeg)

    def run(self):
        seg = self.input()
        seg.open()

        ny = long(seg.shape[1])
        nx = long(seg.shape[2])

        patchShape = [self.patchSize,self.patchSize]

        def extract_patch_statistics_slice(z):
            # 2d blocking representing the patches
            seg_z = seg.read([long(z),0,0],[z+1,ny,nx])
            patches = nifty.tools.blocking( roiBegin = [0L,0L], roiEnd = [ny,nx], blockShape = patchShape )
            # get number of segments for patches in this slice
            n_segs_z = []
            for patch_id in xrange(patches.numberOfBlocks):
                patch = patches.getBlock(patch_id)
                patch_begin, patch_end = patch.begin, patch.end
                patch_slicing = np.s_[patch_begin[0]:patch_end[0],patch_begin[1]:patch_end[1]]
                n_segs_z.append( np.unique(seg_z[patch_slicing]).shape[0] )
            return n_segs_z

        # sequential for debugging
        segs_per_patch = []
        for z in xrange(seg.shape[0]):
            segs_per_patch.extend(extract_patch_statistics_slice(z))

        # parallel
        #with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as executor:
        #    tasks = []
        #    for z in xrange(seg.shape[0]):
        #        tasks.append(executor.submit(extract_patch_statistics_slice, z))
        #    segs_per_patch = []
        #    for fut in tasks:
        #        segs_per_patch.extend(fut.result())

        mean = np.mean(segs_per_patch)
        std  = np.std(segs_per_patch)
        stats = np.array([mean,std])
        print stats
        self.output().write(stats, 'data')


    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "OversegmentationStatistics_%s.h5" % (segFile,) )
        return HDF5DataTarget(save_path)
