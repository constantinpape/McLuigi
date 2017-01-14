# Multicut Pipeline implemented with luigi
# Multicut Solver Tasks

import luigi

import os
import time

from customTargets import HDF5DataTarget

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging
import json

import numpy as np
import nifty

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

class McSolverFusionMoves(luigi.Task):

    problem = luigi.TaskParameter()

    def requires(self):
        return self.problem

    @run_decorator
    def run(self):

        # read the mc parameter
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        mcProblem = self.input()

        g = nifty.graph.UndirectedGraph()

        edgeCosts = mcProblem.read("costs")
        g.deserialize(mcProblem.read("graph"))

        assert g.numberOfEdges == edgeCosts.shape[0]

        obj = nifty.graph.multicut.multicutObjective(g, edgeCosts)

        workflow_logger.info("McSolverFusionMoves: solving multicut problem with %i number of variables" % (g.numberOfNodes,))
        workflow_logger.info("McSolverFusionMoves: using the fusion moves solver from nifty")

        greedy = obj.greedyAdditiveFactory().create(obj)

        t_inf = time.time()
        ret    = greedy.optimize()
        workflow_logger.info("McSolverFusionMoves: energy of the greedy solution %f" % obj.evalNodeLabels(ret) )

        ilpFac = obj.multicutIlpFactory(ilpSolver='cplex',verbose=mc_config["verbose"],
            addThreeCyclesConstraints=True,
            addOnlyViolatedThreeCyclesConstraints=True
        )

        solver = obj.fusionMoveBasedFactory(
            verbose=mc_config["verbose"],
            fusionMove=obj.fusionMoveSettings(mcFactory=ilpFac),
            proposalGen=obj.watershedProposals(sigma=mc_config["sigmaFusion"],seedFraction=mc_config["seedFraction"]),
            numberOfIterations=mc_config["numIt"],
            numberOfParallelProposals=mc_config["numParallelProposals"],
            numberOfThreads=mc_config["numThreadsFusion"],
            stopIfNoImprovement=mc_config["numItStop"],
            fuseN=mc_config["numFuse"],
        ).create(obj)

        if mc_config["verbose"]:
            visitor = obj.multicutVerboseVisitor(1)
            ret = solver.optimize(nodeLabels=ret, visitor=visitor)
        else:
            ret = solver.optimize(nodeLabels=ret)

        t_inf = time.time() - t_inf
        workflow_logger.info("McSolverFusionMoves: inference with fusion move solver in %i s" % (t_inf,))

        # projection to edge result, don't really need it
        # if necessary at some point, make extra task and recover

        #ru = res_node[uv_ids[:,0]]
        #rv = res_node[uv_ids[:,1]]
        #res_edge = ru!=rv

        workflow_logger.info("McSolverFusionMoves: energy of the solution %f" % (obj.evalNodeLabels(ret), ) )

        self.output().write(ret)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "McSolverFusionMoves.h5")
        return HDF5DataTarget( save_path )


# FIXME nifty exact does not work properly for all problems!
class McSolverExact(luigi.Task):

    problem = luigi.TaskParameter()

    def requires(self):
        return self.problem

    @run_decorator
    def run(self):

        # read the mc parameter
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        mcProblem = self.input()

        g = nifty.graph.UndirectedGraph()

        edgeCosts = mcProblem.read("costs")
        g.deserialize(mcProblem.read("graph"))

        assert g.numberOfEdges == edgeCosts.shape[0]

        obj = nifty.graph.multicut.multicutObjective(g, edgeCosts)

        workflow_logger.info("McSolverExact: solving multicut problem with %i number of variables" % (g.numberOfNodes,))
        workflow_logger.info("McSolverExact: using the exact solver from nifty")

        solver = obj.multicutIlpFactory(ilpSolver='cplex',verbose=0,
            addThreeCyclesConstraints=True,
            addOnlyViolatedThreeCyclesConstraints=True
        ).create(obj)

        t_inf = time.time()

        # test time limit
        #visitor = obj.multicutVerboseVisitor(100, 10) # print, timeLimit
        #print "Starting to optimize with time limit"
        #ret = solver.optimize(visitor = visitor)

        ret = solver.optimize()

        t_inf = time.time() - t_inf
        workflow_logger.info("McSolverExact: inference with exact solver in %i s" % (t_inf,))

        # projection to edge result, don't really need it
        # if necessary at some point, make extra task and recover

        #ru = res_node[uv_ids[:,0]]
        #rv = res_node[uv_ids[:,1]]
        #res_edge = ru!=rv

        workflow_logger.info("McSolverExact: energy of the solution %f" % (obj.evalNodeLabels(ret), ) )

        self.output().write(ret)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "McSolverExact.h5")
        return HDF5DataTarget( save_path )
