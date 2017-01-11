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


class OversegmentationPatchStatistics(luigi.Task):

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
        #segs_per_patch = []
        #for z in xrange(seg.shape[0]):
        #    segs_per_patch.extend(extract_patch_statistics_slice(z))

        # parallel
        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as executor:
            tasks = []
            for z in xrange(seg.shape[0]):
                tasks.append(executor.submit(extract_patch_statistics_slice, z))
            segs_per_patch = []
            for fut in tasks:
                segs_per_patch.extend(fut.result())

        mean = np.mean(segs_per_patch)
        std  = np.std(segs_per_patch)

        # calculate histogram to have a closer look at the stats
        n_bins = 16
        histo, bin_edges = np.histogram(segs_per_patch, bins = n_bins)
        bins = np.array([(bin_edges[b] + bin_edges[b+1]) / 2 for b in xrange(n_bins)])

        stats = np.zeros([2*n_bins+2])
        stats[0] = mean
        stats[1] = std
        stats[2:2+n_bins] = histo
        stats[2+n_bins:]  = bins

        self.output().write(stats, 'data')

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "OversegmentationPatchStatistics_%s.h5" % (segFile,) )
        return HDF5DataTarget(save_path)


class OversegmentationSliceStatistics(luigi.Task):

    pathToSeg = luigi.Parameter()
    nBins = luigi.Parameter(default = 16)

    def requires(self):
        return ExternalSegmentation(self.pathToSeg)

    def run(self):
        seg = self.input()
        seg.open()

        ny = long(seg.shape[1])
        nx = long(seg.shape[2])

        def extract_segs_in_slice(z):
            # 2d blocking representing the patches
            seg_z = seg.read([long(z),0,0],[z+1,ny,nx])
            return np.unique(seg_z).shape[0]

        # sequential for debugging
        #segs_per_slice = []
        #for z in xrange(seg.shape[0]):
        #    print z
        #    segs_per_slice.append(extract_segs_in_slice(z))

        # parallel
        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as executor:
            tasks = []
            for z in xrange(seg.shape[0]):
                tasks.append(executor.submit(extract_segs_in_slice, z))
            segs_per_slice = [fut.result() for fut in tasks]

        mean = np.mean(segs_per_slice)
        std  = np.std(segs_per_slice)

        # calculate histogram to have a closer look at the stats
        histo, bin_edges = np.histogram(segs_per_slice, bins = self.nBins)
        #bins = np.array([(bin_edges[b] + bin_edges[b+1]) / 2 for b in xrange(n_bins)])

        out = self.output()
        out.write(mean, 'mean')
        out.write(std, 'std')
        out.write(histo, 'histo')
        out.write(bin_edges, 'bin_edges')

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "OversegmentationSliceStatistics_%s.h5" % (segFile,) )
        return HDF5DataTarget(save_path)


class DefectPatchDetection(luigi.Task):

    pathToSeg = luigi.Parameter()
    patchSize = luigi.Parameter()
    patchOverlap = luigi.Parameter()
    # patches that have below (defectThreshold*mean) number of segment get marked as defected
    # this needs some tuning, 15 % sounds like a reasonable initial guess
    defectThreshold = luigi.Parameter(default = .15)

    def requires(self):
        return {"seg" : ExternalSegmentation(self.pathToSeg),
                "stats" : OversegmentationPatchStatistics(self.pathToSeg,self.patchSize)}

    def run(self):
        inp = self.input()
        seg = inp["seg"]
        stats = inp["stats"].read()

        mean_num_segs = stats[0]

        seg.open()
        out = self.output()
        # TODO compression for nifty hdf5
        out.open(seg.shape)

        ny = long(seg.shape[1])
        nx = long(seg.shape[2])

        patch_shape = [self.patchSize,self.patchSize]
        patch_overlap = [long(self.patchOverlap),long(self.patchOverlap)]

        workflow_logger.info("Running defect detection:")

        def detect_patches_z(z):
            seg_z = seg.read([long(z),0L,0L],[z+1,ny,nx])
            patches = nifty.tools.blocking( roiBegin = [0L,0L], roiEnd = [ny,nx], blockShape = patch_shape )
            # get number of segments for patches in this slice
            n_defected = 0
            for patch_id in xrange(patches.numberOfBlocks):
                patch = patches.getBlockWithHalo(patch_id, patch_overlap).outerBlock
                patch_begin, patch_end = patch.begin, patch.end
                patch_slicing = np.s_[patch_begin[0]:patch_end[0],patch_begin[1]:patch_end[1]]
                # get number of segments in this segment
                n_segments = np.unique(seg_z[patch_slicing]).shape[0]
                # if this number is below threshold, mark the patch as defected
                # TODO need to properly tune this, because false positive hurt really bad...
                # maybe have different numbers for different thresholds to factor in some uncertainties
                if n_segments < self.defectThreshold * mean_num_segs:
                    this_shape = (1,) + tuple(map(lambda x,y:x-y, patch_end, patch_begin))
                    this_patch = np.ones( this_shape , dtype=np.uint8)
                    this_begin = [long(z)] + patch_begin
                    out.write(this_begin, this_patch)
                    n_defected += 1
            return n_defected

        # non parallel for debugging
        #defects_per_slice = []
        #for z in xrange(seg.shape[0]):
        #    print z
        #    defects_per_slice.append(detect_patches_z(z))

        with futures.ThreadPoolExecutor(max_workers = PipelineParameter().nThreads) as executor:
            tasks = []
            for z in xrange(seg.shape[0]):
                tasks.append(executor.submit(detect_patches_z,z))
            defects_per_slice = [fut.result() for fut in tasks]

        # log the defects
        workflow_logger.info("Total number of defected patches: %i" % np.sum(defects_per_slice))
        for z in xrange(seg.shape[0]):
            if defects_per_slice[z] > 0:
                workflow_logger.info("Slice %i has %i defected patches." % (z,defects_per_slice[z]))

        out.close()

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "DefectPatchDetection_%s.h5" % (segFile,) )
        return HDF5VolumeTarget(save_path, dtype = 'uint8')


class DefectSliceDetection(luigi.Task):

    pathToSeg = luigi.Parameter()
    # we consider a slice as defected, when it is in a bin smaller than binThreshold
    binThreshold = luigi.IntParameter(default = 1)

    def requires(self):
        return {"seg" : ExternalSegmentation(self.pathToSeg),
                "stats" : OversegmentationSliceStatistics(self.pathToSeg)}

    def run(self):
        inp = self.input()
        seg = inp["seg"]
        bin_edges = inp["stats"].read("bin_edges")

        threshold = bin_edges[self.binThreshold]

        seg.open()
        out = self.output()
        # TODO compression for nifty hdf5
        out.open(seg.shape)

        ny = long(seg.shape[1])
        nx = long(seg.shape[2])

        workflow_logger.info("Running defect detection:")

        slice_shape = (1L,ny,nx)
        defect_mask = np.ones(slice_shape, dtype = np.uint8)

        def detect_defected_slice(z):
            slice_begin = [long(z),0L,0L]
            seg_z = seg.read(slice_begin,[z+1,ny,nx])
            # get number of segments for patches in this slice
            n_segs = np.unique(seg_z).shape[0]
            # threshold for a defected slice
            if n_segs < threshold:
                out.write(slice_begin, defect_mask)
                return True
            return False

        # non parallel for debugging
        #defect_indications = []
        #for z in xrange(seg.shape[0]):
        #    print z
        #    defect_indications.append(detect_defected_slice(z))

        with futures.ThreadPoolExecutor(max_workers = PipelineParameter().nThreads) as executor:
            tasks = []
            for z in xrange(seg.shape[0]):
                tasks.append(executor.submit(detect_defected_slice,z))
            defect_indications = [fut.result() for fut in tasks]

        # log the defects
        workflow_logger.info("Total number of defected slices: %i" % np.sum(defect_indications))
        for z in xrange(seg.shape[0]):
            if defect_indications[z]:
                workflow_logger.info("Slice %i is defected." % (z,))

        out.close()

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "DefectSliceDetection_%s.h5" % (segFile,) )
        return HDF5VolumeTarget(save_path, dtype = 'uint8')

# TODO implement
# map defect patches to overlapping superpixels
class DefectPatchesToSegments(luigi.Task):

    def requires(self):
        pass


# TODO implement
# modify the rag to isolate defected superpixels
# introduce skip edges over the defected superpixels in z
class ModifiedRag(luigi.Task):

    def requires(self):
        pass


# TODO implement
# calculate features for the skip edges
class SkipFeatures(luigi.Task):

    def requires(self):
        pass


# TODO implement
# set the xy edges connecting defected superpixel to non-defected superpixels to be repulsive
class ModifyWeights(luigi.Task):

    def requires(self):
        pass
