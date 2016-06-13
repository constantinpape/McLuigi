# Multicut Pipeline implemented with luigi
# Workflow Tasks

import luigi

from MulticutSolverTasks import MCProblem, MCSSolverOpengmExact, MCSSolverOpengmFusionMoves
from BlockwiseMulticutTasks import BlockwiseMulticutSolver
from DataTasks import RegionAdjacencyGraph
from PipelineParameter import PipelineParameter
from CustomTargets import HDF5Target

import logging
import json
import os

import numpy as np
import vigra


class MulticutSegmentation(luigi.Task):

    PathToSeg = luigi.Parameter()
    PathToRF  = luigi.Parameter()

    def requires(self):
        return { "MCRes" : MCSSolverOpengmExact( MCProblem(self.PathToSeg, self.PathToRF) ),
                "RAG" : RegionAdjacencyGraph(self.PathToSeg) }

    def run(self):

        # get the projection of a multicut result to the segmentation
        rag = self.input()["RAG"].read()
        mc_res = self.input()["MCRes"].read()

        assert mc_res.shape[0] == rag.nodeNum

        self.output().write( rag.projectLabelsToBaseGraph(mc_res.astype(np.uint32)) )


    def output(self):
        save_path = os.path.join( PipelineParameter().cache,
                "MCSegmentation_" + os.path.split(self.PathToSeg)[1][:-3] + ".h5" )
        return HDF5Target( save_path )


class BlockwiseMulticutSegmentation(luigi.Task):

    PathToSeg = luigi.Parameter()
    PathToRF  = luigi.Parameter()

    def requires(self):
        return { "MCRes" : BlockwiseMulticutSolver( self.PathToSeg, self.PathToRF ),
                "RAG" : RegionAdjacencyGraph(self.PathToSeg) }

    def run(self):

        # get the projection of a multicut result to the segmentation
        rag = self.input()["RAG"].read()
        mc_res = self.input()["MCRes"].read()

        assert mc_res.shape[0] == rag.nodeNum

        self.output().write( rag.projectLabelsToBaseGraph(mc_res.astype(np.uint32)) )


    def output(self):
        save_path = os.path.join( PipelineParameter().cache,
                "BlockwiseMCSegmentation_" + os.path.split(self.PathToSeg)[1][:-3] + ".h5" )
        return HDF5Target( save_path )



