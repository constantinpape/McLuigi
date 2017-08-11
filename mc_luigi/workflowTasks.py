from __future__ import division, print_function

# Multicut Pipeline implemented with luigi
# Workflow Tasks

import luigi

from multicutProblemTasks import MulticutProblem

from multicutSolverTasks import McSolverFusionMoves  # ,McSolverExact
from blockwiseMulticutTasks import BlockwiseMulticutSolver
from blockwiseBaselineTasks import SubblockSegmentations, BlockwiseOverlapSolver
from blockwiseBaselineTasks import BlockwiseStitchingSolver, BlockwiseMulticutStitchingSolver, NoStitchingSolver
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation
from customTargets import HDF5VolumeTarget
from defectDetectionTasks import DefectSliceDetection
from skeletonTasks import ResolveCandidates

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator, get_replace_slices

import logging
import os

import numpy as np
import vigra

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


class SegmentationWorkflow(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToClassifier  = luigi.Parameter()
    dtype = luigi.Parameter(default='uint32')

    def requires(self):
        raise AttributeError(
            "SegmentationWorkflow should never be called, \
            call MulticutSegmentation or BlockwiseMulticutSegmentation instead!"
        )

    @run_decorator
    def run(self):

        inp = self.input()
        rag = inp["rag"].read()
        mc_nodes = inp["mc_nodes"].read().astype(self.dtype)
        seg = inp["seg"]

        seg.open()
        shape = seg.shape()
        out = self.output()
        out.open(shape)

        ## TODO: this is only temporary for the sampleD experiments
        ## where we don't need the segmentation (and writing it takes quite long....)
        #seg.close()
        #out.close()
        #quit()

        workflow_logger.info("SegmentationWorkflow: Projecting node result to segmentation.")
        self._project_result_to_segmentation(rag, mc_nodes, out)

        if PipelineParameter().defectPipeline:
            workflow_logger.info("SegmentationWorkflow: Postprocessing defected slices.")
            self._postprocess_defected_slices(inp, out)

        out.close()
        seg.close()

    def _project_result_to_segmentation(self, rag, mc_nodes, out):
        assert mc_nodes.shape[0] == rag.numberOfNodes
        # get rid of 0 because we don't want it as segment label because it is reserved for the ignore label
        mc_nodes, _, _ = vigra.analysis.relabelConsecutive(mc_nodes, start_label=0, keep_zeros=False)
        if np.dtype(self.dtype) != np.dtype(mc_nodes.dtype):
            self.dtype = mc_nodes.dtype
        nrag.projectScalarNodeDataToPixels(rag, mc_nodes, out.get(), 5)  # TODO investigate number of threads here

    def _postprocess_defected_slices(self, inp, out):

        defect_slices_path = inp['defect_slices'].path
        shape = out.shape()
        defected_slices = vigra.readHDF5(defect_slices_path, 'defect_slices')

        # we only replace slices if we actually have completely defected slices
        if not defected_slices.size:
            workflow_logger.info("SegmentationWorkflow: No completely defected slices found, doing nothing.")
            return

        replace_slice = get_replace_slices(defected_slices, shape)

        for z in defected_slices:
            replace_z = replace_slice[z]
            workflow_logger.info("SegmentationWorkflow: replacing defected slice %i by %i" % (z, replace_z))
            out.write(
                [z, 0L, 0L],
                out.read([replace_z, 0L, 0L], [replace_z + 1, shape[1], shape[2]])
            )


    # TODO expand the segmentation by offset !
    def _expand_offset(self):
        pass

    def output(self):
        raise AttributeError(
            "SegmentationWorkflow should never be called, \
            call MulticutSegmentation or BlockwiseMulticutSegmentation instead!"
        )


class MulticutSegmentation(SegmentationWorkflow):

    def requires(self):
        return_tasks = {
            "mc_nodes": McSolverFusionMoves(MulticutProblem(self.pathToSeg, self.pathToClassifier)),
            "rag": StackedRegionAdjacencyGraph(self.pathToSeg),
            "seg": ExternalSegmentation(self.pathToSeg)
        }
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectSliceDetection(self.pathToSeg)
        return return_tasks

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "MulticutSegmentation_%s.h5" % (
                "modified" if PipelineParameter().defectPipeline else "standard",
            )
        )
        return HDF5VolumeTarget(save_path, self.dtype, compression=PipelineParameter().compressionLevel)


class BlockwiseMulticutSegmentation(SegmentationWorkflow):

    numberOfLevels = luigi.IntParameter(default=1)

    def requires(self):
        return_tasks = {
            "mc_nodes": BlockwiseMulticutSolver(
                self.pathToSeg,
                MulticutProblem(self.pathToSeg, self.pathToClassifier),
                self.numberOfLevels
            ),
            "rag": StackedRegionAdjacencyGraph(self.pathToSeg),
            "seg": ExternalSegmentation(self.pathToSeg)
        }
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectSliceDetection(self.pathToSeg)
        return return_tasks

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "BlockwiseMulticutSegmentation_L%i_%s_%s_%s.h5" % (
                self.numberOfLevels,
                '_'.join(map(str, PipelineParameter().multicutBlockShape)),
                '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
                "modified" if PipelineParameter().defectPipeline else "standard",
            )
        )
        return HDF5VolumeTarget(save_path, self.dtype, compression=PipelineParameter().compressionLevel)


class BlockwiseStitchingSegmentation(SegmentationWorkflow):

    numberOfLevels = luigi.IntParameter(default=1)
    boundaryBias = luigi.FloatParameter(default=.5)

    def requires(self):
        return_tasks = {
            "mc_nodes": BlockwiseStitchingSolver(
                self.pathToSeg,
                MulticutProblem(self.pathToSeg, self.pathToClassifier),
                self.numberOfLevels,
                self.boundaryBias
            ),
            "rag": StackedRegionAdjacencyGraph(self.pathToSeg),
            "seg": ExternalSegmentation(self.pathToSeg)
        }
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectSliceDetection(self.pathToSeg)
        return return_tasks

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "BlockwiseStitchingSegmentation_L%i_%s_%s_%s_%.2f.h5" % (
                self.numberOfLevels,
                '_'.join(map(str, PipelineParameter().multicutBlockShape)),
                '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
                "modified" if PipelineParameter().defectPipeline else "standard",
                self.boundaryBias
            )
        )
        return HDF5VolumeTarget(save_path, self.dtype, compression=PipelineParameter().compressionLevel)


class BlockwiseMulticutStitchingSegmentation(SegmentationWorkflow):

    numberOfLevels = luigi.IntParameter(default=1)

    def requires(self):
        return_tasks = {
            "mc_nodes": BlockwiseMulticutStitchingSolver(
                self.pathToSeg,
                MulticutProblem(self.pathToSeg, self.pathToClassifier),
                self.numberOfLevels
            ),
            "rag": StackedRegionAdjacencyGraph(self.pathToSeg),
            "seg": ExternalSegmentation(self.pathToSeg)
        }
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectSliceDetection(self.pathToSeg)
        return return_tasks

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "BlockwiseMulticutStitchingSegmentation_L%i_%s_%s_%s.h5" % (
                self.numberOfLevels,
                '_'.join(map(str, PipelineParameter().multicutBlockShape)),
                '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
                "modified" if PipelineParameter().defectPipeline else "standard",
            )
        )
        return HDF5VolumeTarget(save_path, self.dtype, compression=PipelineParameter().compressionLevel)


class BlockwiseOverlapSegmentation(SegmentationWorkflow):

    numberOfLevels = luigi.IntParameter(default=1)

    def requires(self):
        return_tasks = {
            "mc_nodes": BlockwiseOverlapSolver(
                self.pathToSeg,
                MulticutProblem(self.pathToSeg, self.pathToClassifier),
                self.numberOfLevels
            ),
            "rag": StackedRegionAdjacencyGraph(self.pathToSeg),
            "seg": ExternalSegmentation(self.pathToSeg)
        }
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectSliceDetection(self.pathToSeg)
        return return_tasks

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "BlockwiseOverlapSegmentation_L%i_%s_%s_%s_%.2f.h5" % (
                self.numberOfLevels,
                '_'.join(map(str, PipelineParameter().multicutBlockShape)),
                '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
                "modified" if PipelineParameter().defectPipeline else "standard",
                PipelineParameter().overlapThreshold
            )
        )
        return HDF5VolumeTarget(save_path, self.dtype, compression=PipelineParameter().compressionLevel)


class NoStitchingSegmentation(SegmentationWorkflow):

    numberOfLevels = luigi.IntParameter(default=1)

    def requires(self):
        return_tasks = {
            "mc_nodes": NoStitchingSolver(
                self.pathToSeg,
                MulticutProblem(self.pathToSeg, self.pathToClassifier),
                self.numberOfLevels
            ),
            "rag": StackedRegionAdjacencyGraph(self.pathToSeg),
            "seg": ExternalSegmentation(self.pathToSeg)
        }
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectSliceDetection(self.pathToSeg)
        return return_tasks

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "NoStitchingSegmentation_L%i_%s_%s_%s.h5" % (
                self.numberOfLevels,
                '_'.join(map(str, PipelineParameter().multicutBlockShape)),
                '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
                "modified" if PipelineParameter().defectPipeline else "standard"
            )
        )
        return HDF5VolumeTarget(save_path, self.dtype, compression=PipelineParameter().compressionLevel)


# TODO
# -> make different skeleton methods accessible
class ResolvedSegmentation(SegmentationWorkflow):

    fragmentationPath = luigi.Parameter()
    segmentationPath = luigi.Parameter()
    fragmentClassifierPath = luigi.Parameter()
    weight = luigi.Parameter(default=1.)
    numberOfLevels = luigi.Parameter(default=1)

    def requires(self):
        return {
            "mc_nodes": ResolveCandidates(
                self.fragmentationPath,
                self.segmentationPath,
                self.fragmentClassifierPath,
                self.weight,
                self.numberOfLevels
            ),
            "rag": StackedRegionAdjacencyGraph(self.fragmentationPath),
            "seg": ExternalSegmentation(self.fragmentationPath)
        }

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "ResolvedSegmentation_%s.h5" % (
                "modified" if PipelineParameter().defectPipeline else "standard",
            )
        )
        return HDF5VolumeTarget(save_path, self.dtype, compression=PipelineParameter().compressionLevel)


class SubblockSegmentationWorkflow(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToClassifier  = luigi.Parameter()
    numberOfLevels = luigi.IntParameter(default=1)
    dtype = luigi.Parameter(default='uint32')

    def requires(self):
        return SubblockSegmentations(
            self.pathToSeg,
            MulticutProblem(self.pathToSeg, self.pathToClassifier),
            self.numberOfLevels
        )

    def run(self):
        pass

    def output(self):
        pass
