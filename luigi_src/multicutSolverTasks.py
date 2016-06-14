# Multicut Pipeline implemented with luigi
# Multicut Solver Tasks

import luigi

from dataTasks import RegionAdjacencyGraph
from learningTasks import EdgeProbabilitiesFromExternalRF
from miscTasks import EdgeIndications
from customTargets import HDF5Target

from pipelineParameter import PipelineParameter
from toolsLuigi import config_logger

import logging
import json

import numpy as np
import vigra
import os
import time

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

class MCSSolverOpengmFusionMoves(luigi.Task):

    Problem = luigi.TaskParameter()

    def requires(self):
        return Problem

    def run(self):
        import opengm

        # read the mc parameter
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        mc_problem = self.input().read()

        uv_ids = mc_problem[:,:2]
        edge_costs = mc_problem[:,2]

        n_var = uv_ids.max() + 1

        workflow_logger.info("Solving MC Problem with " + str(n_var) + " number of variables")

        # set up the opengm model
        states = np.ones(n_var) * n_var
        gm = opengm.gm(states)

        # potts model
        potts_shape = [n_var, n_var]

        potts = opengm.pottsFunctions(potts_shape, np.zeros_like( edge_costs ), edge_costs )

        # potts model to opengm function
        fids_b = gm.addFunctions(potts)

        gm.addFactors(fids_b, uv_ids)

        pparam = opengm.InfParam(seedFraction = mc_config["SeedFraction"])
        parameter = opengm.InfParam(generator = 'randomizedWatershed',
                                    proposalParam = pparam,
                                    numStopIt = mc_config["NumItStop"],
                                    numIt = mc_configonfig["NumIt"])

        inf = opengm.inference.IntersectionBased(gm, parameter=parameter)

        t_inf = time.time()
        inf.infer()
        t_inf = time.time() - t_inf

        workflow_loggerinfo("Inference with fusin moves solver in " + str(t_inf) + " s")

        res_node = inf.arg()

        # projection to edge result, don't really need it
        # if necessary at some point, make extra task and recover

        #ru = res_node[uv_ids[:,0]]
        #rv = res_node[uv_ids[:,1]]
        #res_edge = ru!=rv

        e_glob = gm.evaluate(res_node)

        workflow_logger.info("Energy of the solution " + str(e_glob) )

        self.output().write(res_node)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "MCSSolverOpengmFusionMoves.h5")
        return HDF5Target( save_path )


class MCSSolverOpengmExact(luigi.Task):

    Problem = luigi.TaskParameter()

    def requires(self):
        return self.Problem

    def run(self):
        import opengm

        mc_problem = self.input().read()

        uv_ids = mc_problem[:,:2]
        edge_costs = mc_problem[:,2]

        n_var = uv_ids.max() + 1

        workflow_logger.info("Solving MC Problem with " + str(n_var) + " number of variables")

        # set up the opengm model
        states = np.ones(n_var) * n_var
        gm = opengm.gm(states)

        # potts model
        potts_shape = [n_var, n_var]

        potts = opengm.pottsFunctions(potts_shape, np.zeros_like( edge_costs ), edge_costs )

        # potts model to opengm function
        fids_b = gm.addFunctions(potts)

        gm.addFactors(fids_b, uv_ids)

        # the workflow, we use
        wf = "(IC)(CC-IFD)"

        # TODO incorporate verbosity into logging
        param = opengm.InfParam( workflow = wf, verbose = False,
                verboseCPLEX = False, numThreads = 4 )

        inf = opengm.inference.Multicut(gm, parameter=param)
        t_inf = time.time()
        inf.infer()
        t_inf = time.time() - t_inf

        workflow_logger.info("Inference with exact solver in " + str(t_inf) + " s")

        res_node = inf.arg()

        # projection to edge result, don't really need it
        # if necessary at some point, make extra task and recover

        #ru = res_node[uv_ids[:,0]]
        #rv = res_node[uv_ids[:,1]]
        #res_edge = ru!=rv

        e_glob = gm.evaluate(res_node)

        workflow_logger.info("Energy of the solution " + str(e_glob) )

        self.output().write(res_node)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "MCSSolverOpengmExact.h5")
        return HDF5Target( save_path )


# get weights and uvids of the MC problem
class MCProblem(luigi.Task):

    PathToSeg = luigi.Parameter()
    PathToRF  = luigi.Parameter()

    def requires(self):
        return { "EdgeProbs" : EdgeProbabilitiesFromExternalRF(self.PathToRF),
                "RAG" : RegionAdjacencyGraph(self.PathToSeg),
                "EdgeIndications" : EdgeIndications(self.PathToSeg) }

    def run(self):

        # read the mc parameter
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        rag = self.input()["RAG"].read()

        # get the uvids
        uv_ids = np.sort( rag.uvIds(), axis = 1 )

        assert uv_ids.max() + 1 == rag.nodeNum, str(uv_ids.max() + 1)  + " , " + str(rag.nodeNum)

        # get the edge costs

        # scale the probabilities
        # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
        probs = self.input()["EdgeProbs"].read()

        p_min = 0.001
        p_max = 1. - p_min
        probs = (p_max - p_min) * probs + p_min

        beta = mc_config["Beta"]

        # probabilities to energies, second term is boundary bias
        edge_costs = np.log( (1. - probs) / probs ) + np.log( (1. - beta) / beta )

        # weight edge costs
        weighting_scheme = mc_config["WeightingScheme"]
        weight           = mc_config["Weight"]

        workflow_logger.info("Weighting edge costs with scheme " + weighting_scheme + " and weight " + str(weight) )
        if weighting_scheme == "z":
            edges_size = rag.edgeLengths()
            edge_indications = self.input()["EdgeIndications"].read()
            assert edges_size.shape[0] == edge_costs.shape[0]
            assert edge_indications.shape[0] == edge_costs.shape[0]

            # z - edges are indicated with 0 !
            z_max = float( np.max( edges_size[edge_indications == 0] ) )
            # we only weight the z edges !
            w = weight * edges_size[edge_indications == 0] / z_max
            edge_costs[edge_indications == 0] = np.multiply(w, edge_costs[edge_indications == 0])

        elif weighting_scheme == "xyz":
            edges_size =rag.edgeLengths()
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
