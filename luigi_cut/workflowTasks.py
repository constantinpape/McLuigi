# Multicut Pipeline implemented with luigi
# Workflow Tasks

import luigi

from multicutSolverTasks import McProblem, McSolverExact, McSolverFusionMoves
from blockwiseMulticutTasks import BlockwiseMulticutSolver
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation
from customTargets import HDF5VolumeTarget

from pipelineParameter import PipelineParameter
from toolsLuigi import config_logger

import logging
import json
import os

import numpy as np
import nifty


class MulticutSegmentation(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToRF  = luigi.Parameter()

    def requires(self):
        return { "McNodes" : McSolverFusionMoves( McProblem(self.pathToSeg, self.pathToRF) ),
                "Rag" : StackedRegionAdjacencyGraph(self.pathToSeg),
                "Seg" : ExternalSegmentation(self.pathToSeg)}

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

        segOut = self.output()
        segOut.open(seg.shape)

        nifty.graph.rag.projectScalarNodeDataToPixels(rag, mcNodes, segOut.get() ) # nWorkers = -1, could also set this...


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "MulticutSegmentation.h5" )
        return HDF5VolumeTarget( save_path, np.uint32 )


class BlockwiseMulticutSegmentation(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToRF  = luigi.Parameter()

    numberOfLevels = luigi.Parameter(default = 2)

    def requires(self):
        return { "McNodes" : BlockwiseMulticutSolver( self.pathToSeg, self.pathToRF, self.numberOfLevels ),
                "Rag" : StackedRegionAdjacencyGraph(self.pathToSeg),
                "Seg" : ExternalSegmentation(self.pathToSeg)}

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

        segOut = self.output()
        segOut.open(seg.shape)

        nifty.graph.rag.projectScalarNodeDataToPixels(rag, mcNodes, segOut.get() ) # nWorkers = -1, could also set this...


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "BlockwiseMulticutSegmentation.h5" )
        return HDF5VolumeTarget( save_path, np.uint32 )
