# Multicut Pipeline implemented with luigi
# Workflow Tasks

import luigi

from multicutProblemTasks import MulticutProblem

from multicutSolverTasks import McSolverExact, McSolverFusionMoves
from blockwiseMulticutTasks import BlockwiseMulticutSolver
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation
from customTargets import HDF5VolumeTarget

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging
import json
import os

import numpy as np
import nifty


class MulticutSegmentation(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathsToClassifier  = luigi.ListParameter()
    dtype = luigi.Parameter(default = 'uint32')

    def requires(self):
        return { "McNodes" : McSolverFusionMoves(
                    MulticutProblem(self.pathToSeg, self.pathsToClassifier) ),
                "Rag" : StackedRegionAdjacencyGraph(self.pathToSeg),
                "Seg" : ExternalSegmentation(self.pathToSeg)}

    @run_decorator
    def run(self):

        inp = self.input()
        rag = inp["Rag"].read()
        mcNodes = inp["McNodes"].read()
        seg = inp["Seg"]

        seg.open()
        shape = seg.shape

        assert mcNodes.shape[0] == rag.numberOfNodes

        # get rid of 0 because we don't want it as segment label because it is reserved for the ignore label
        if 0 in mcNodes:
            mcNodes += 1

        if np.dtype(self.dtype) != np.dtype(mcNodes.dtype):
            self.dtype = mcNodes.dtype

        segOut = self.output()
        segOut.open(seg.shape)

        nifty.graph.rag.projectScalarNodeDataToPixels(rag, mcNodes, segOut.get() ) # nWorkers = -1, could also set this...


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "MulticutSegmentation.h5" )
        return HDF5VolumeTarget( save_path, self.dtype, compression = PipelineParameter().compressionLevel)


class BlockwiseMulticutSegmentation(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathsToClassifier  = luigi.ListParameter()

    dtype = luigi.Parameter(default = 'uint32')
    numberOfLevels = luigi.IntParameter(default = 2)

    def requires(self):
        return { "McNodes" : BlockwiseMulticutSolver( self.pathToSeg,
                    MulticutProblem(self.pathToSeg, self.pathsToClassifier),
                    self.numberOfLevels ),
                "Rag" : StackedRegionAdjacencyGraph(self.pathToSeg),
                "Seg" : ExternalSegmentation(self.pathToSeg)}

    @run_decorator
    def run(self):

        inp = self.input()
        rag = inp["Rag"].read()
        mcNodes = inp["McNodes"].read()
        seg = inp["Seg"]

        seg.open()
        shape = seg.shape

        assert mcNodes.shape[0] == rag.numberOfNodes

        # get rid of 0 because we don't want it as segment label because it is reserved for the ignore label
        if 0 in mcNodes:
            mcNodes += 1

        if np.dtype(self.dtype) != np.dtype(mcNodes.dtype):
            self.dtype = mcNodes.dtype

        segOut = self.output()
        segOut.open(seg.shape)

        nifty.graph.rag.projectScalarNodeDataToPixels(rag, mcNodes, segOut.get() ) # nWorkers = -1, could also set this...
        # TODO if PipelineParameter().defectPipeline call PostprocessDefectSlices once implemented

    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "BlockwiseMulticutSegmentation_%s.h5" % ("modifed" if PipelineParameter().defectPipeline else "standard",) )
        return HDF5VolumeTarget( save_path, self.dtype, compression = PipelineParameter().compressionLevel )
