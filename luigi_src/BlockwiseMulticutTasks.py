# Multicut Pipeline implemented with luigi
# Multicut Solver Tasks

import luigi

from PipelineParameter import PipelineParameter
from DataTasks import RegionAdjacencyGraph
from MulticutSolverTasks import MCProblem, MCSSolverOpengmFusionMoves, MCSSolverOpengmExact

import logging
import json

import numpy as np
import vigra

# TODO would be better to let the scheduler handle the parallelisation

class BlockwiseMulticutSolver(luigi.Task):

    PathToSeg = luigi.Parameter()
    PathToRF  = luigi.Parameter()

    def requires(self):
        return BlockwiseSubProblems( self.PathToSeg, self.PathToRF )

    def run(self):
        pass

    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "MCBlockwise.h5")
        return HDF5Target( save_path )


class BlockwiseSubProblems(luigi.Task):

    def requires(self):
