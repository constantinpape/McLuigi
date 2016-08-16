# Multicut Pipeline implemented with luigi
# Multicut Solver Tasks

import luigi

import os
import time

from dataTasks import StackedRegionAdjacencyGraph
from learningTasks import EdgeProbabilitiesFromExternalRF
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

class MCSSolverNiftyFusionMoves(luigi.Task):

    problem = luigi.TaskParameter()

    def requires(self):
        return self.problem

    def run(self):

        # read the mc parameter
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        mcProblem = self.input().read()

        uvIds     = mcProblem[:,:2]
        edgeCosts = mcProblem[:,2]
        nVariables = uvIds.max() + 1

        g = nifty.graph.UndirectedGraph(int(nVariables))
        g.insertEdges(uvIds)

        assert g.numberOfEdges == edgeCosts.shape[0]

        obj = nifty.graph.multicut.multicutObjective(g, edgeCosts)

        workflow_logger.info("Solving multicut problem with %i number of variables" % (nVariables,))
        workflow_logger.info("Using the fusion moves solver from nifty")

        greedy = obj.greedyAdditiveFactory().create(obj)

        t_inf = time.time() - t_inf
        ret    = greedy.optimize()
        workflow_logger.info("Energy of the greedy solution" +  str(obj.evalNodeLabels(ret)) )

        ilpFac = obj.multicutIlpFactory(ilpSolver='cplex',verbose=0,
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

        ret = solver.optimize(nodeLabels=ret)
        t_inf = time.time() - t_inf
        workflow_loggerinfo("Inference with exact solver in %i s" % (t_inf,))

        # projection to edge result, don't really need it
        # if necessary at some point, make extra task and recover

        #ru = res_node[uv_ids[:,0]]
        #rv = res_node[uv_ids[:,1]]
        #res_edge = ru!=rv

        workflow_logger.info("Energy of the solution %i" % (obj.evalNodeLabels(ret), ) )

        self.output().write(ret)



    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "MCSSolverOpengmFusionMoves.h5")
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

        # TODO use the parameters for initialising the solver!
        mcProblem = self.input().read()

        uvIds     = mcProblem[:,:2]
        edgeCosts = mcProblem[:,2]
        nVariables = uvIds.max() + 1

        g = nifty.graph.UndirectedGraph(int(nVariables))
        g.insertEdges(uvIds)

        assert g.numberOfEdges == edgeCosts.shape[0]

        obj = nifty.graph.multicut.multicutObjective(g, edgeCosts)

        workflow_logger.info("Solving multicut problem with %i number of variables" % (nVariables,))
        workflow_logger.info("Using the exact solver from nifty")

        solver = obj.multicutIlpFactory(ilpSolver='cplex',verbose=0,
            addThreeCyclesConstraints=True,
            addOnlyViolatedThreeCyclesConstraints=True
        ).create(obj)

        t_inf = time.time()
        ret = solver.optimize()
        t_inf = time.time() - t_inf
        workflow_loggerinfo("Inference with exact solver in %i s" % (t_inf,))

        # projection to edge result, don't really need it
        # if necessary at some point, make extra task and recover

        #ru = res_node[uv_ids[:,0]]
        #rv = res_node[uv_ids[:,1]]
        #res_edge = ru!=rv

        workflow_logger.info("Energy of the solution %i" % (obj.evalNodeLabels(ret), ) )

        self.output().write(ret)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "MCSSolverOpengmFusionMoves.h5")
        return HDF5Target( save_path )


# get weights and uvids of the MC problem
class McProblem(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToRF  = luigi.Parameter()

    def requires(self):
        return { "EdgeProbabilities" : EdgeProbabilitiesFromExternalRF(self.pathToRF),
                "Rag" : StackedRegionAdjacencyGraph(self.pathToSeg) }

    def run(self):

        # read the mc parameter
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        inp = self.input()
        rag = inp["Rag"].read()
        probs = inp["EdgeProbabilities"].read()

        uvIds = rag.uvIds

        # scale the probabilities
        # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
        p_min = 0.001
        p_max = 1. - p_min
        probs = (p_max - p_min) * probs + p_min

        beta = mc_config["Beta"]

        # probabilities to energies, second term is boundary bias
        edge_costs = np.log( (1. - probs) / probs ) + np.log( (1. - beta) / beta )

        # weight edge costs
        weighting_scheme = mc_config["WeightingScheme"]
        weight           = mc_config["Weight"]

        # TODO need the size of the edges for weighting!
        workflow_logger.info("Weighting edge costs with scheme " + weighting_scheme + " and weight " + str(weight) )
        if weighting_scheme == "z":
            edges_size = rag.numberOfEdges
            edge_indications = self.input()["EdgeIndications"].read()
            assert edges_size.shape[0] == edge_costs.shape[0]
            assert edge_indications.shape[0] == edge_costs.shape[0]

            # z - edges are indicated with 0 !
            z_max = float( np.max( edges_size[edge_indications == 0] ) )
            # we only weight the z edges !
            w = weight * edges_size[edge_indications == 0] / z_max
            edge_costs[edge_indications == 0] = np.multiply(w, edge_costs[edge_indications == 0])

        elif weighting_scheme == "xyz":
            edges_size = rag.numberOfEdges
            edge_indications = self.input()["EdgeIndications"].read()
            assert edges_size.shape[0] == edge_costs.shape[0]
            assert edge_indications.shape[0] == edge_costs.shape[0]

            z_max  = float( np.max( edges_size[edge_indications == 0] ) )
            xy_max = float( np.max( edges_size[edge_indications == 1] ) )
            w_z  = weight * edges_size[edge_indications == 0] / z_max
            w_xy = weight * edges_size[edge_indications == 1] / xy_max
            edge_costs[edge_indications == 0] = np.multiply(w_z, edge_costs[edge_indications == 0])
            edge_costs[edge_indications == 1] = np.multiply(w_xy, edge_costs[edge_indications == 1])

        elif weighting_scheme == "all":
            edges_size =rag.edgeLengths()
            edges_size = rag.edgeLengths()
            assert edges_size.shape[0] == edge_costs.shape[0]

            edge_max  = float( np.max( edges_size) )
            w  = weight * edges_size / edge_max
            edge_costs = np.multiply(w, edge_costs)

        assert edge_costs.shape[0] == uv_ids.shape[0]
        assert np.isfinite( edge_costs.min() ), str(edge_costs.min())
        assert np.isfinite( edge_costs.max() ), str(edge_costs.max())

        # write concatenation of uvids and edge costs
        self.output().write( np.concatenate( [uv_ids, edge_costs[:,None]], axis = 1 ) )


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "MCProblem.h5" )
        return HDF5Target( save_path )
