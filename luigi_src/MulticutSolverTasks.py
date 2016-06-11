# Multicut Pipeline implemented with luigi
# Multicut Solver Tasks

import luigi

from DataTasks import RegionAdjacencyGraph

import logging

import numpy as np
import vigra

#class BlockwiseSolver(lugi.task):

class MCSSolverOpengmFusionMoves(luigi.task):

    PathToSeg = luigi.Parameter()
    MCParameter = luigi.DictParameter()
    FeatureTasks = luigi.ListParameter()
    SolverParameter = lugi.DictParameter(significant = False)

    def requires(self):
        return MCProblem(self.PathToSeg, self.MCParameter)

    def run(self):
        import opengm


    def output(self):

class MCSSolverOpengmExact(luigi.task):

    MCParameter = luigi.DictParameter()
    SolverParameter = lugi.DictParameter(significant = False)

    def requires(self):
        return MCWeights(self.MCParameter)

    def run(self):
        import opengm

    def output(self):


# get weights and size of the MC problem
class MCProblem(luigi.task):

    PathToSeg = luigi.Parameter()
    MCParameter = luigi.DictParameter()
    FeatureTasks = luigi.ListParameter()

    def requires(self):
        # TODO implement Edge Indications Task
        return [EdgeProbabilities(self.FeatureTasks), RegionAdjacencyGraph(self.PathToSeg),
                EdgeIndications(self.PathToSeg)]

    def run(self):

        rag = self.input()[1].read()

        # get the problem size

        # get the edge costs

        # scale the probabilities
        # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
        probs = self.input()[0].read()
        p_min = 0.001
        p_max = 1. - p_min
        edge_probs = (p_max - p_min) * probs + p_min
        beta = self.MCParameter["BetaLocalEdges"]

        # probabilities to energies, second term is boundary bias
        edge_costs = np.log( (1. - probs) / probs ) + np.log( (1. - beta) / beta )

        # weight edge costs
        weighting_scheme = self.MCParameter["WeightingSchemeLocal"]
        weight = self.MCParameter["CostWeightLocal"]
        # TODO own loglevel for pipeline related stuff
        logging.info("Weighting edge costs with scheme " + weighting_scheme + " and weight " + weight)
        if exp_params.weighting_scheme == "z":
            edges_size = rag.edgeLengths()
            edge_indications = self.input().read()
            assert edges_size.shape[0] == edge_costs.shape[0]
            assert edge_indications.shape[0] == edge_costs.shape[0]

            # z - edges are indicated with 0 !
            z_max = float( np.max( edges_size[edge_indications == 0] ) )
            # we only weight the z edges !
            w = weight * edges_size[edge_indications == 0] / z_max
            edge_costs[edge_indications == 0] = np.multiply(w, edge_costs[edge_indications == 0])

       elif exp_params.weighting_scheme == "xyz":
            edges_size =rag.edgeLengths()
            edge_indications = self.input().read()
            assert edges_size.shape[0] == edge_costs.shape[0]
            assert edge_indications.shape[0] == edge_costs.shape[0]

            z_max  = float( np.max( edges_size[edge_indications == 0] ) )
            xy_max = float( np.max( edges_size[edge_indications == 1] ) )
            w_z  = weight * edges_size[edge_indications == 0] / z_max
            w_xy = weight * edges_size[edge_indications == 1] / xy_max
            edge_costs[edge_indications == 0] = np.multiply(w_z, edge_costs[edge_indications == 0])
            edge_costs[edge_indications == 1] = np.multiply(w_xy, edge_costs[edge_indications == 1])

        elif exp_params.weighting_scheme == "all":
            edges_size = rag.edgeLengths()
            assert edges_size.shape[0] == edge_costs.shape[0]

            edge_max  = float( np.max( edges_size) )
            w  = weight * edges_size / edge_max
            edge_costs = np.multiply(w, edge_costs)

        # TODO write output


    def output(self):
        # TODO more meaningful caching name
        save_path = os.path.join( PipelineParameter().cache,
                os.path.split(self.PathToSeg)[1] )
        return HDF5Target( save_path )
