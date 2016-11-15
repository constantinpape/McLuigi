# Multicut Pipeline implemented with luigi
# Multicut Solver Tasks

import luigi

import os
import time

from dataTasks import StackedRegionAdjacencyGraph
from learningTasks import EdgeProbabilities
from customTargets import HDF5DataTarget

from pipelineParameter import PipelineParameter
from toolsLuigi import config_logger

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

        workflow_logger.info("Solving multicut problem with %i number of variables" % (g.numberOfNodes,))
        workflow_logger.info("Using the fusion moves solver from nifty")

        greedy = obj.greedyAdditiveFactory().create(obj)

        t_inf = time.time()
        ret    = greedy.optimize()
        workflow_logger.info("Energy of the greedy solution" +  str(obj.evalNodeLabels(ret)) )

        ilpFac = obj.multicutIlpFactory(ilpSolver='cplex',verbose=0,
            addThreeCyclesConstraints=True,
            addOnlyViolatedThreeCyclesConstraints=True
        )

        solver = obj.fusionMoveBasedFactory(
            #verbose=mc_config["verbose"],
            verbose=2,
            fusionMove=obj.fusionMoveSettings(mcFactory=ilpFac),
            proposalGen=obj.watershedProposals(sigma=mc_config["sigmaFusion"],seedFraction=mc_config["seedFraction"]),
            numberOfIterations=mc_config["numIt"],
            numberOfParallelProposals=mc_config["numParallelProposals"],
            numberOfThreads=mc_config["numThreadsFusion"],
            stopIfNoImprovement=mc_config["numItStop"],
            fuseN=mc_config["numFuse"],
        ).create(obj)

        ## test time limit
        #visitor = obj.multicutVerboseVisitor(1, 1) # print, timeLimit
        #print "Starting to optimize with time limit"
        #ret = solver.optimize(nodeLabels=ret, visitor=visitor)

        ret = solver.optimize(nodeLabels=ret)

        t_inf = time.time() - t_inf
        workflow_logger.info("Inference with fusion move solver in %i s" % (t_inf,))

        # projection to edge result, don't really need it
        # if necessary at some point, make extra task and recover

        #ru = res_node[uv_ids[:,0]]
        #rv = res_node[uv_ids[:,1]]
        #res_edge = ru!=rv

        workflow_logger.info("Energy of the solution %i" % (obj.evalNodeLabels(ret), ) )

        self.output().write(ret)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "McSolverFusionMoves.h5")
        return HDF5DataTarget( save_path )


# FIXME nifty exact does not work responsibly for all problems!
class McSolverExact(luigi.Task):

    problem = luigi.TaskParameter()

    def requires(self):
        return self.problem

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

        workflow_logger.info("Solving multicut problem with %i number of variables" % (g.numberOfNodes,))
        workflow_logger.info("Using the exact solver from nifty")

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
        workflow_logger.info("Inference with exact solver in %i s" % (t_inf,))

        # projection to edge result, don't really need it
        # if necessary at some point, make extra task and recover

        #ru = res_node[uv_ids[:,0]]
        #rv = res_node[uv_ids[:,1]]
        #res_edge = ru!=rv

        workflow_logger.info("Energy of the solution %i" % (obj.evalNodeLabels(ret), ) )

        self.output().write(ret)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "McSolverExact.h5")
        return HDF5DataTarget( save_path )


# get weights and uvids of the MC problem
class McProblem(luigi.Task):

    pathToSeg = luigi.Parameter()
    # this can either contain a single path (classifier trained for xy - and z - edges jointly)
    # or two paths (classfier trained for xy - edges + classifier trained for z - edges separately)
    pathsToClassifier  = luigi.ListParameter()

    def requires(self):
        return { "EdgeProbabilities" : EdgeProbabilities(self.pathToSeg, self.pathsToClassifier),
                "Rag" : StackedRegionAdjacencyGraph(self.pathToSeg) }

    def run(self):

        # read the mc parameter
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        inp = self.input()
        rag = inp["Rag"].read()
        edgeCosts = inp["EdgeProbabilities"].read()

        uvIds = rag.uvIds()

        # scale the probabilities
        # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
        p_min = 0.001
        p_max = 1. - p_min
        edgeCosts = (p_max - p_min) * edgeCosts + p_min

        beta = mc_config["beta"]

        # probabilities to energies, second term is boundary bias
        edgeCosts = np.log( (1. - edgeCosts) / edgeCosts ) + np.log( (1. - beta) / beta )

        # weight edge costs
        weighting_scheme = mc_config["weightingScheme"]
        weight           = mc_config["weight"]

        workflow_logger.info("Weighting edge costs with scheme " + weighting_scheme + " and weight " + str(weight) )
        if weighting_scheme == "z":
            edgeLens = rag.edgeLengths()
            assert edgeLens.shape[0] == edgeCosts.shape[0], str(edgeLens.shape[0]) + " , " + str(edgeCosts.shape[0])

            edgeTransition = rag.totalNumberOfInSliceEdges

            z_max = float( np.max( edgeLens[edgeTransition:] ) )
            # we only weight the z edges !
            w = weight * edgeLens[edgeTransition:] / z_max
            edgeCosts[edgeTransition:] = np.multiply(w, edgeCosts[edgeTransition:])

        elif weighting_scheme == "xyz":
            edgeLens = rag.edgeLengths()
            assert edgesLen.shape[0] == edgeCosts.shape[0]

            edgeTransition = rag.totalNumberOfInSliceEdges

            z_max = float( np.max( edgeLens[edgeTransition:] ) )
            xy_max = float( np.max( edgeLens[:edgeTransition] ) )
            w_z = weight * edgeLens[edgeTransition:] / z_max
            w_xy = weight * edgeLens[:edgeTransition] / xy_max
            edgeCosts[edgeTransition:] = np.multiply(w_z,  edgeCosts[edgeTransition:])
            edgeCosts[:edgeTransition] = np.multiply(w_xy, edgeCosts[:edgeTransition])

        elif weighting_scheme == "all":
            edgeLens = rag.edgeLengths()
            assert edgesLen.shape[0] == edgeCosts.shape[0]

            edge_max  = float( np.max( edgesLens) )
            w  = weight * edgeLens / edge_max
            edgeCosts = np.multiply(w, edgeCosts)

        assert edgeCosts.shape[0] == uvIds.shape[0]
        assert np.isfinite( edgeCosts.min() ), str(edgeCosts.min())
        assert np.isfinite( edgeCosts.max() ), str(edgeCosts.max())

        nVariables = uvIds.max() + 1
        g = nifty.graph.UndirectedGraph(int(nVariables))
        g.insertEdges(uvIds)

        # write concatenation of uvids and edge costs
        out = self.output()

        assert g.numberOfEdges == edgeCosts.shape[0]
        out.write( g.serialize(), "graph" )
        out.write( edgeCosts, "costs")


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "McProblem.h5" )
        return HDF5DataTarget( save_path )
