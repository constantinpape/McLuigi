# Multicut Pipeline implemented with luigi
# Workflow Tasks

import luigi

from multicutSolverTasks import MCProblem, MCSSolverOpengmExact, MCSSolverOpengmFusionMoves
from blockwiseMulticutTasks import BlockwiseMulticutSolver
from dataTasks import RegionAdjacencyGraph
from customTargets import HDF5Target

from pipelineParameter import PipelineParameter
from toolsLuigi import config_logger

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

        # get rid of 0 because we don't want it as segment label because it is reserved for the ignore label
        if 0 in mc_res:
            mc_res += 1

        self.output().write( rag.projectLabelsToBaseGraph(mc_res.astype(np.uint32)) )


    def output(self):
        #save_path = os.path.join( PipelineParameter().cache,
        #        "MCSegmentation_" + os.path.split(self.PathToSeg)[1][:-3] + ".h5" )
        #return HDF5Target( save_path )
        save_path = os.path.join( PipelineParameter().cache, "MulticutSegmentation.h5" )
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

        # get rid of 0 because we don't want it as segment label because it is reserved for the ignore label
        if 0 in mc_res:
            mc_res += 1

        self.output().write( rag.projectLabelsToBaseGraph(mc_res.astype(np.uint32)) )


    def output(self):
        #save_path = os.path.join( PipelineParameter().cache,
        #        "BlockwiseMulitcutSegmentation_" + os.path.split(self.PathToSeg)[1][:-3] + ".h5" )
        #return HDF5Target( save_path )
        save_path = os.path.join( PipelineParameter().cache, "BlockwiseMulticutSegmentation.h5" )
        return HDF5Target( save_path )



