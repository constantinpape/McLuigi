from __future__ import division, print_function

# Multicut Pipeline implemented with luigi
# Workflow Tasks

import luigi

from .multicutProblemTasks import MulticutProblem
from .multicutSolverTasks import McSolverFusionMoves
from .blockwiseMulticutTasks import BlockwiseMulticutSolver
from .blockwiseBaselineTasks import SubblockSegmentations, BlockwiseOverlapSolver
from .blockwiseBaselineTasks import BlockwiseStitchingSolver, BlockwiseMulticutStitchingSolver, NoStitchingSolver
from .dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation
from .customTargets import VolumeTarget
from .defectHandlingTasks import DefectsToNodes
# from .skeletonTasks import ResolveCandidates

from .pipelineParameter import PipelineParameter
from .tools import config_logger, run_decorator, get_replace_slices

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
    keyToSeg = luigi.Parameter(default='data')
    dtype = luigi.Parameter(default='uint64')
    savePath = luigi.Parameter(default='')
    saveKey = luigi.Parameter(default='data')

    def requires(self):
        raise AttributeError(
            "SegmentationWorkflow should never be called, \
            call MulticutSegmentation or BlockwiseMulticutSegmentation instead!"
        )

    @run_decorator
    def run(self):

        inp = self.input()
        rag = inp["rag"].read()
        mc_nodes = inp["mc_nodes"].read().astype(self.dtype, copy=False)
        seg = inp["seg"]

        seg.open(self.keyToSeg)
        shape = seg.shape(self.keyToSeg)

        chunks = (1, min(1024, shape[1]), min(1024, shape[2]))
        out = self.output()
        out.open(self.saveKey, shape=shape, chunks=chunks, dtype=self.dtype)

        workflow_logger.info("SegmentationWorkflow: Projecting node result to segmentation.")
        self._project_result_to_segmentation(rag, mc_nodes, out)

        if PipelineParameter().defectPipeline:
            workflow_logger.info("SegmentationWorkflow: Postprocessing defected slices.")
            self._postprocess_defected_slices(inp, out)

        seg.close()
        out.close()

    def _project_result_to_segmentation(self, rag, mc_nodes, out):
        assert mc_nodes.shape[0] == rag.numberOfNodes
        mc_nodes, _, _ = vigra.analysis.relabelConsecutive(mc_nodes,
                                                           start_label=1,
                                                           keep_zeros=False)
        # if we have an ignore label, set it's node value to zero
        if PipelineParameter().ignoreSegLabel != -1:
            workflow_logger.info("SegmentationWorkflow: Setting node values for ignore seg value: %i to 0."
                                 % PipelineParameter().ignoreSegLabel)
            mc_nodes[PipelineParameter().ignoreSegLabel] = 0

        if np.dtype(self.dtype) != np.dtype(mc_nodes.dtype):
            self.dtype = mc_nodes.dtype
        nrag.projectScalarNodeDataToPixels(rag,
                                           mc_nodes,
                                           out.get(self.saveKey),
                                           PipelineParameter().nThreads)

    def _postprocess_defected_slices(self, inp, out):

        defect_slices_path = inp['defect_slices'].path
        shape = out.shape(self.saveKey)
        defected_slices = vigra.readHDF5(defect_slices_path, 'defect_slices')

        # we only replace slices if we actually have completely defected slices
        if not defected_slices.size:
            workflow_logger.info("SegmentationWorkflow: No completely defected slices found, doing nothing.")
            return

        replace_slice = get_replace_slices(defected_slices, shape)

        for z in defected_slices:
            replace_z = replace_slice[z]
            workflow_logger.info("SegmentationWorkflow: replacing defected slice %i by %i" % (z, replace_z))
            out.write([z, 0, 0],
                      out.read([replace_z, 0, 0], [replace_z + 1, shape[1], shape[2]]),
                      key=self.saveKey)

    def output(self):
        raise AttributeError(
            "SegmentationWorkflow should not be called, \
            call MulticutSegmentation or BlockwiseMulticutSegmentation instead!"
        )


class MulticutSegmentation(SegmentationWorkflow):

    def requires(self):
        return_tasks = {"mc_nodes": McSolverFusionMoves(MulticutProblem(self.pathToSeg,
                                                                        self.pathToClassifier)),
                        "rag": StackedRegionAdjacencyGraph(self.pathToSeg),
                        "seg": ExternalSegmentation(self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectsToNodes(self.pathToSeg)
        return return_tasks

    def output(self):
        if self.savePath is '':
            save_path = os.path.join(PipelineParameter().cache,
                                     "MulticutSegmentation_%s.h5" % (
                                         "modified" if PipelineParameter().defectPipeline else "standard",))
        else:
            save_path = self.savePath
        return VolumeTarget(save_path)


class BlockwiseMulticutSegmentation(SegmentationWorkflow):

    numberOfLevels = luigi.IntParameter(default=1)

    def requires(self):
        return_tasks = {"mc_nodes": BlockwiseMulticutSolver(self.pathToSeg,
                                                            MulticutProblem(self.pathToSeg,
                                                                            self.pathToClassifier,
                                                                            keyToSeg=self.keyToSeg),
                                                            self.numberOfLevels,
                                                            keyToSeg=self.keyToSeg),
                        "rag": StackedRegionAdjacencyGraph(self.pathToSeg, self.keyToSeg),
                        "seg": ExternalSegmentation(self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectsToNodes(self.pathToSeg)
        return return_tasks

    def output(self):
        if self.savePath is '':
            save_path = os.path.join(
                PipelineParameter().cache,
                "BlockwiseMulticutSegmentation_L%i_%s_%s_%s" % (
                    self.numberOfLevels,
                    '_'.join(map(str, PipelineParameter().multicutBlockShape)),
                    '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
                    "modified" if PipelineParameter().defectPipeline else "standard",
                )
            )
            save_path += VolumeTarget.file_ending()
        else:
            save_path = self.savePath
        return VolumeTarget(save_path)


class BlockwiseStitchingSegmentation(SegmentationWorkflow):

    numberOfLevels = luigi.IntParameter(default=1)
    boundaryBias = luigi.FloatParameter(default=.5)

    def requires(self):
        return_tasks = {"mc_nodes": BlockwiseStitchingSolver(self.pathToSeg,
                                                             MulticutProblem(self.pathToSeg,
                                                                             self.pathToClassifier,
                                                                             keyToSeg=self.keyToSeg),
                                                             self.numberOfLevels,
                                                             boundaryBia=self.boundaryBias,
                                                             keyToSeg=self.keyToSeg),
                        "rag": StackedRegionAdjacencyGraph(self.pathToSeg, self.keyToSeg),
                        "seg": ExternalSegmentation(self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectsToNodes(self.pathToSeg)
        return return_tasks

    def output(self):
        if self.savePath is '':
            save_path = os.path.join(
                PipelineParameter().cache,
                "BlockwiseStitchingSegmentation_L%i_%s_%s_%s_%.2f" % (
                    self.numberOfLevels,
                    '_'.join(map(str, PipelineParameter().multicutBlockShape)),
                    '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
                    "modified" if PipelineParameter().defectPipeline else "standard",
                    self.boundaryBias
                )
            )
            save_path += VolumeTarget.file_ending()
        else:
            save_path = self.savePath
        return VolumeTarget(save_path)


class BlockwiseMulticutStitchingSegmentation(SegmentationWorkflow):

    numberOfLevels = luigi.IntParameter(default=1)

    def requires(self):
        return_tasks = {"mc_nodes": BlockwiseMulticutStitchingSolver(self.pathToSeg,
                                                                     MulticutProblem(self.pathToSeg,
                                                                                     self.pathToClassifier,
                                                                                     keyToSeg=self.keyToSeg),
                                                                     self.numberOfLevels),
                        "rag": StackedRegionAdjacencyGraph(self.pathToSeg, self.keyToSeg),
                        "seg": ExternalSegmentation(self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectsToNodes(self.pathToSeg)
        return return_tasks

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "BlockwiseMulticutStitchingSegmentation_L%i_%s_%s_%s" % (
                self.numberOfLevels,
                '_'.join(map(str, PipelineParameter().multicutBlockShape)),
                '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
                "modified" if PipelineParameter().defectPipeline else "standard",
            )
        )
        save_path += VolumeTarget.file_ending()
        return VolumeTarget(save_path)


class BlockwiseOverlapSegmentation(SegmentationWorkflow):

    numberOfLevels = luigi.IntParameter(default=1)

    def requires(self):
        return_tasks = {"mc_nodes": BlockwiseOverlapSolver(self.pathToSeg,
                                                           MulticutProblem(self.pathToSeg,
                                                                           self.pathToClassifier,
                                                                           keyToSeg=self.keyToSeg),
                                                           self.numberOfLevels),
                        "rag": StackedRegionAdjacencyGraph(self.pathToSeg, self.keyToSeg),
                        "seg": ExternalSegmentation(self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectsToNodes(self.pathToSeg)
        return return_tasks

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "BlockwiseOverlapSegmentation_L%i_%s_%s_%s_%.2f" % (
                self.numberOfLevels,
                '_'.join(map(str, PipelineParameter().multicutBlockShape)),
                '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
                "modified" if PipelineParameter().defectPipeline else "standard",
                PipelineParameter().overlapThreshold
            )
        )
        save_path += VolumeTarget.file_ending()
        return VolumeTarget(save_path)


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
            return_tasks["defect_slices"] = DefectsToNodes(self.pathToSeg)
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
        return VolumeTarget(save_path)


# TODO
# -> make different skeleton methods accessible
# class ResolvedSegmentation(SegmentationWorkflow):
#
#     fragmentationPath = luigi.Parameter()
#     segmentationPath = luigi.Parameter()
#     fragmentClassifierPath = luigi.Parameter()
#     weight = luigi.Parameter(default=1.)
#     numberOfLevels = luigi.Parameter(default=1)
#
#     def requires(self):
#         return {
#             "mc_nodes": ResolveCandidates(
#                 self.fragmentationPath,
#                 self.segmentationPath,
#                 self.fragmentClassifierPath,
#                 self.weight,
#                 self.numberOfLevels
#             ),
#             "rag": StackedRegionAdjacencyGraph(self.fragmentationPath),
#             "seg": ExternalSegmentation(self.fragmentationPath)
#         }
#
#     def output(self):
#         save_path = os.path.join(
#             PipelineParameter().cache,
#             "ResolvedSegmentation_%s.h5" % (
#                 "modified" if PipelineParameter().defectPipeline else "standard",
#             )
#         )
#         return VolumeTarget(save_path)


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
