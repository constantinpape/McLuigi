# Multicut Pipeline implemented with luigi
# Tasks for providing the input data

import luigi
from customTargets import HDF5VolumeTarget, StackedRagTarget, FolderTarget

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging

import numpy as np
import vigra
import os
import h5py
import time

from concurrent import futures
import subprocess

# TODO benchmark and debug alternative impl
from wsdt_impl import compute_wsdt_segmentation
# from wsdt import wsDtSegmentation as compute_wsdt_segmentation

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


# FIXME this won't work for larger data.
# -> affinity prediction is not parallelized yet
# -> not everything is done out of core (affinity reshaping)
# TODO check plans of speeding up gunpowder predictions with Jan

# first predict affinities with mala,
# then reshape them properly (xy, z affinities)
# TODO do reshaping in gunpowder output node already
class AffinityPrediction(luigi.Task):

    rawPath = luigi.Parameter()

    def requires(self):
        return InputData(self.rawPath)

    @run_decorator
    def run(self):

        # TODO change this if we have correct gunpowder output directly and don't need
        # postprocessing steps any longer
        out = self.output()
        out_path = os.path.join(out.path, 'prediction_tmp.h5')

        workflow_logger.info("AffinityPrediction: writing net predictions to %s" % out_path)

        t_pred = time.time()
        self._predict_affinities(out_path)
        workflow_logger.info("AffinityPrediction: prediction took %f s" % (time.time() - t_pred,))

        t_pp = time.time()
        self._postprocess_affinities(out_path)
        workflow_logger.info("AffinityPrediction: postprocessing took %f s" % (time.time() - t_pp,))

    def _predict_affinities(self, out_path):

        net_path = PipelineParameter().netArchitecturePath
        assert os.path.exists(net_path), net_path
        net_weights = PipelineParameter().netWeightsPath
        assert os.path.exists(net_weights), net_weights

        workflow_logger.info(
            "AffinityPrediction: starting prediction for raw data in %s with net from %s %s"
            % (self.rawPath, net_path, net_weights)
        )

        scripts_path = os.path.split(__file__)[0]
        affinity_exe_path = os.path.join(scripts_path, 'affinity_prediction.sh')
        prediction_script = os.path.join(scripts_path, 'affinity_prediction.py')
        assert os.path.exists(affinity_exe_path), affinity_exe_path
        assert os.path.exists(prediction_script), prediction_script

        command_line = [
            'bash', affinity_exe_path,
            prediction_script, self.rawPath,
            out_path, net_path,
            net_weights, str(PipelineParameter().netGpuId)
        ]

        workflow_logger.info(
            "AffinityPrediction: calling prediction script with command: %s" % str(command_line)
        )
        subprocess.call(command_line)

    def _postprocess_affinities(self, out_path):

        assert os.path.exists(out_path), out_path
        raw_name = os.path.split(self.rawPath)[1]

        # crop the padding context from the raw data and save it!
        with h5py.File(out_path) as f_aff:

            ds_aff = f_aff['volumes/predicted_affs']
            aff_shape = ds_aff.shape[1:]

            inp = self.input()
            inp.open()

            # if for some reason we already have offsets, they need to be over-written
            if inp.has_offsets():
                inp.setOffsets([0, 0, 0], [0, 0, 0])
                workflow_logger.info("AffinityPrediction: overwriting old raw offsets")

            raw_shape = inp.shape()

            offset = np.array(list(raw_shape)) - np.array(list(aff_shape))
            offset /= 2

            # write the raw data offsets to the nifty h5 array
            inp.setOffsets(offset.tolist(), offset.tolist())
            assert tuple(inp.shape()) == aff_shape, "%s, %s" % (str(inp.shape()), str(aff_shape))
            inp.close()
            workflow_logger.info(
                "AffinityPrediction: cropping cnn context %s from raw data" % str(offset)
            )

            # save the xy-affinities and the z-affinities seperately
            affinity_folder = self.output().path
            save_path_xy = os.path.join(affinity_folder, '%s_affinities_xy.h5' % raw_name[:-3])
            save_path_z  = os.path.join(affinity_folder, '%s_affinities_z.h5' % raw_name[:-3])
            with h5py.File(save_path_xy) as f_xy, h5py.File(save_path_z) as f_z:

                # TODO compress ?, uint8 ?
                ds_xy = f_xy.create_dataset('data', shape=aff_shape, chunks=(1, 512, 512), dtype='float32')
                ds_z = f_z.create_dataset('data', shape=aff_shape, chunks=(1, 512, 512), dtype='float32')

                # TODO this could be parallelized and done in a out-of-core fashion !
                ds_xy[:] = (ds_aff[1, :] + ds_aff[2, :]) / 2.
                ds_z[:] = (ds_aff[0, :])

        # remover the temp file
        os.remove(out_path)

    def output(self):
        raw_prefix = os.path.split(self.rawPath)[1][:-3]
        affinity_folder = os.path.join(PipelineParameter().cache, '%s_affinities' % raw_prefix)
        return FolderTarget(affinity_folder)


class WsdtSegmentation(luigi.Task):
    """
    Task for generating segmentation via wsdt.
    """

    pathToProbabilities = luigi.Parameter()
    dtype = luigi.Parameter(default='uint32')

    def requires(self):
        """
        Dependencies:
        """
        return InputData(self.pathToProbabilities)

    # TODO enable 3d wsdt for isotropic ppl
    # TODO we could integrate the (statistics based) defect detection directly in here
    # TODO parallelisation is really weird / slow, investigate !
    # -> check if same strange behaviour on sirherny
    # if not, probably nifty-package is responsible -> hdf5 parallel read/write ?! (libhdf5-serial ?)
    # -> if this is true, also other applications should be horribly slow
    @run_decorator
    def run(self):
        pmap = self.input()
        pmap.open()
        shape = pmap.shape()
        out  = self.output()
        out.open(shape, pmap.chunk_shape())

        self._run_wsdt2d_standard(pmap, out, shape)

        out.close()
        pmap.close()

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

            print "Slice", z, "/", shape[0]
            sliceStart = [z, 0, 0]
            sliceStop  = [z + 1, shape[1], shape[2]]
            pmap_z = pmap.read(sliceStart, sliceStop).squeeze()

            if invert:
                pmap_z = 1. - pmap_z

            seg, max_z = compute_wsdt_segmentation(pmap_z, threshold, sig_seeds, min_seg)
            out.write(sliceStart, seg[None, :, :])
            return max_z + 1

        n_workers = PipelineParameter().nThreads

        t_ws = time.time()
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(segment_slice, z) for z in xrange(shape[0])]
            offsets = [future.result() for future in tasks]
        workflow_logger.info(
            "WsdtSegmentation: Running watershed took: %f s" % (time.time() - t_ws)
        )

        # accumulate the offsets for each slice
        offsets = np.roll(offsets, 1)
        offsets[0] = 0
        offsets = np.cumsum(offsets)

        if offsets[-1] >= 4294967295 and self.dtype == np.dtype('uint32'):
            print "WARNING: UINT32 overflow!"
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
            tasks = [executor.submit(add_offset, z, offsets[z]) for z in xrange(shape[0])]
            [t.result() for t in tasks]
        workflow_logger.info("WsdtSegmentation: Adding offsets took %f s" % (time.time() - t_off))
        return True

    # TODO if we integrate defects / 3d ws reflect this in the caching name
    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "WsdtSegmentation_%s.h5" % os.path.split(self.pathToProbabilities)[1][:-3]
        )
        return HDF5VolumeTarget(save_path, self.dtype, compression=PipelineParameter().compressionLevel)


###########################
# Tasks for providing data
###########################


class InputData(luigi.Task):
    """
    Task for loading external input data, e.g. raw data or probability maps.
    For HDF5 input.
    """

    path = luigi.Parameter()
    key  = luigi.Parameter(default="data")
    # the dtype, should either be float32 or uint8
    dtype = luigi.Parameter(default="float32")

    # if the input data is not existing, this is an affinity map task
    # and we need to predict the affinities first
    def requires(self):
        if not os.path.exists(self.path):

            # find the corresponding raw data from the inputs
            raw_prefix = os.path.split(self.path)[1].split('_')[:-2]
            raw_prefix = '_'.join(raw_prefix) + '.h5'
            inputs = PipelineParameter().inputs['data']
            found_raw = False
            for in_path in inputs:
                if os.path.split(in_path)[1] == raw_prefix:
                    raw_path = in_path
                    found_raw = True
                    break
            if not found_raw:
                raise RuntimeError("Couldn't find raw path for requested affinity map: %s" % self.path)

            return AffinityPrediction(raw_path)

    def run(self):
        pass

    def output(self):
        """
        Returns the target output.

        :return: Target output
        :rtype: object( :py:class: HDF5Target)
        """

        if os.path.exists(self.path):
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

    def run(self):
        pass

    def output(self):
        """
        Returns the target output.

        :return: Target output
        :rtype: object( :py:class: HDF5Target)
        """

        if os.path.exists(self.path):
            with h5py.File(self.path, 'r') as f:
                assert self.key in f.keys(), self.key + " , " + str(f.keys())
                dset = f[self.key]

                if np.dtype(self.dtype) != np.dtype(dset.dtype):
                    workflow_logger.debug("ExternalSegmentation: loading data from %s" % self.path)
                    workflow_logger.debug(
                        "ExternalSegmentation: changing dtype from %s to %s" % (self.dtype, dset.dtype)
                    )
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

        # FIXME this will be problematic for ignore labels that are not 0
        # easiest fix is to reimplement 'relabelConsecutive' based on 'replace_from_dict'
        # and allow to keep arbitrary values fixed

        gt_labeled, _, _ = vigra.analysis.relabelConsecutive(
            gt.read([0, 0, 0], gt.shape()),
            start_label=1,
            keep_zeros=True
        )

        out = self.output()
        out.open(gt.shape())
        out.write([0, 0, 0], gt_labeled)

        gt.close()
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
