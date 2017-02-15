# Multicut Pipeline implemented with luigi
# Workflow Tasks

import luigi

from multicutProblemTasks import MulticutProblem

from multicutSolverTasks import McSolverExact, McSolverFusionMoves
from blockwiseMulticutTasks import BlockwiseMulticutSolver
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation
from customTargets import HDF5VolumeTarget
from defectDetectionTasks import DefectSliceDetection

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging
import json
import os

import numpy as np
import vigra
import nifty

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

class SegmentationWorkflow(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathsToClassifier  = luigi.ListParameter()
    dtype = luigi.Parameter(default = 'uint32')

    def requires(self):
        raise AttributeError("Segmentation Workflow should never be called, call Multicut Segmentation or BlockwiseMulticutSegmentation instead!")

    @run_decorator
    def run(self):

        inp = self.input()
        rag = inp["rag"].read()
        mc_nodes = inp["mc_nodes"].read().astype(self.dtype)
        seg = inp["seg"]

        seg.open()
        shape = seg.shape
        out = self.output()
        out.open(seg.shape)

        workflow_logger.info("SegmentationWorkflow: Projecting node result to segmentation.")
        self._project_result_to_segmentation(rag, mc_nodes, out)

        out.close()
        seg.close()

        if PipelineParameter().defectPipeline:
            workflow_logger.info("SegmentationWorkflow: Postprocessing defected slices.")
            self._postprocess_defected_slices(inp, out)


    def _project_result_to_segmentation(self, rag, mc_nodes, out):
        assert mc_nodes.shape[0] == rag.numberOfNodes
        # get rid of 0 because we don't want it as segment label because it is reserved for the ignore label
        mc_nodes, _, _ = vigra.analysis.relabelConsecutive(mc_nodes, start_label = 0, keep_zeros = False)
        if np.dtype(self.dtype) != np.dtype(mc_nodes.dtype):
                self.dtype = mc_nodes.dtype
        nifty.graph.rag.projectScalarNodeDataToPixels(rag, mc_nodes, out.get(), 5 )


    def _postprocess_defected_slices(self, inp, out):
        import h5py

        defect_slices_path = inp['defect_slices'].path
        shape = out.shape
        defected_slices = vigra.readHDF5(defect_slices_path, 'defect_slices')

        # find consecutive slices with defects
        consecutive_defects = np.split(defected_slices, np.where(np.diff(defected_slices) != 1)[0] + 1)
        # find the replace slices for defected slices
        replace_slice = {}
        for consec in consecutive_defects:
            if len(consec) == 1:
                z = consec[0]
                replace_slice[z] = z - 1 if z > 0 else 1
            elif len(consec) == 2:
                z0, z1 = consec[0], consec[1]
                replace_slice[z0] = z0 - 1 if z0 > 0 else 2
                replace_slice[z1] = z1 + 1 if z1 < shape[0] - 1 else z1 - 2
            elif len(consec) == 3:
                z0, z1, z2 = consec[0], consec[1], consec[2]
                replace_slice[z0] = z0 - 1 if z0 > 0 else 3
                replace_slice[z1] = z1 - 2 if z1 > 1 else 3
                replace_slice[z2] = z2 + 1 if z2 < shape[0] - 1 else z2 - 3
            elif len(consec) == 3:
                z0, z1, z2, z3 = consec[0], consec[1], consec[2], consec[3]
                replace_slice[z0] = z0 - 1 if z0 > 0 else 4
                replace_slice[z1] = z1 - 2 if z1 > 1 else 4
                replace_slice[z2] = z2 + 1 if z2 < shape[0] - 1 else z2 - 3
                replace_slice[z3] = z3 + 2 if z3 < shape[0] - 1 else z3 - 4
            else:
                raise RuntimeError("Postprocessing is not implemented for more than 4 consecutively defected slices. Go and clean your data!")

        # FIXME strange nifty bugs, that's why we use h5py here
        # post-process defected slices by replacement
        with h5py.File(out.path) as f:
            out_ds = f['data']
            for z in defected_slices:
                replace_z = replace_slice[z]
                workflow_logger.info("SegmentationWorkflow: replacing defected slice %i by %i" % (z,replace_z))
                replace = out_ds[replace_z]
                out_ds[z] = replace

    def output(self):
        raise AttributeError("Segmentation Workflow should never be called, call Multicut Segmentation or BlockwiseMulticutSegmentation instead!")


class MulticutSegmentation(SegmentationWorkflow):

    def requires(self):
        return_tasks = { "mc_nodes" : McSolverFusionMoves(MulticutProblem(self.pathToSeg, self.pathsToClassifier) ),
                "rag" : StackedRegionAdjacencyGraph(self.pathToSeg),
                "seg" : ExternalSegmentation(self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectSliceDetection(self.pathToSeg)
        return return_tasks

    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "MulticutSegmentation.h5" )
        return HDF5VolumeTarget( save_path, self.dtype, compression = PipelineParameter().compressionLevel)


class BlockwiseMulticutSegmentation(SegmentationWorkflow):

    numberOfLevels = luigi.IntParameter(default = 2)

    def requires(self):
        return_tasks = {"mc_nodes" : BlockwiseMulticutSolver( self.pathToSeg,
                                     MulticutProblem(self.pathToSeg, self.pathsToClassifier),
                                     self.numberOfLevels ),
                        "rag" : StackedRegionAdjacencyGraph(self.pathToSeg),
                        "seg" : ExternalSegmentation(self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectSliceDetection(self.pathToSeg)
        return return_tasks

    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "BlockwiseMulticutSegmentation_%s.h5" % ("modifed" if PipelineParameter().defectPipeline else "standard",) )
        return HDF5VolumeTarget( save_path, self.dtype, compression = PipelineParameter().compressionLevel )
