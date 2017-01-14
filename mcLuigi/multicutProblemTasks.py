# Multicut Pipeline implemented with luigi
# Multicut Problem Tasks

import luigi

import os
import time

from dataTasks import StackedRegionAdjacencyGraph
from learningTasks import EdgeProbabilities
from customTargets import HDF5DataTarget
from defectHandlingTasks import ModifiedAdjacency

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging
import json

import numpy as np
import nifty

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


def weight_edge_costs(edge_costs, rag, weighting_scheme, weight ):

    workflow_logger.info("Weighting edge costs with scheme " + weighting_scheme + " and weight " + str(weight) )
    if weighting_scheme == "z":
        edgeLens = rag.edgeLengths()
        assert edgeLens.shape[0] == edge_costs.shape[0], str(edgeLens.shape[0]) + " , " + str(edge_costs.shape[0])

        edgeTransition = rag.totalNumberOfInSliceEdges

        z_max = float( np.max( edgeLens[edgeTransition:] ) )
        # we only weight the z edges !
        w = weight * edgeLens[edgeTransition:] / z_max
        edge_costs[edgeTransition:] = np.multiply(w, edge_costs[edgeTransition:])

    elif weighting_scheme == "xyz":
        edgeLens = rag.edgeLengths()
        assert edgesLen.shape[0] == edge_costs.shape[0]

        edgeTransition = rag.totalNumberOfInSliceEdges

        z_max = float( np.max( edgeLens[edgeTransition:] ) )
        xy_max = float( np.max( edgeLens[:edgeTransition] ) )
        w_z = weight * edgeLens[edgeTransition:] / z_max
        w_xy = weight * edgeLens[:edgeTransition] / xy_max
        edge_costs[edgeTransition:] = np.multiply(w_z,  edge_costs[edgeTransition:])
        edge_costs[:edgeTransition] = np.multiply(w_xy, edge_costs[:edgeTransition])

    elif weighting_scheme == "all":
        edgeLens = rag.edgeLengths()
        assert edgesLen.shape[0] == edge_costs.shape[0]

        edge_max  = float( np.max( edgesLens) )
        w  = weight * edgeLens / edge_max
        edge_costs = np.multiply(w, edge_costs)

    return edge_costs


# get weights and uvids of the MC problem
class StandardMulticutProblem(luigi.Task):

    pathToSeg = luigi.Parameter()
    # this can either contain a single path (classifier trained for xy - and z - edges jointly)
    # or two paths (classfier trained for xy - edges + classifier trained for z - edges separately)
    pathsToClassifier  = luigi.ListParameter()

    def requires(self):
        return { "EdgeProbabilities" : EdgeProbabilities(self.pathToSeg, self.pathsToClassifier),
                "Rag" : StackedRegionAdjacencyGraph(self.pathToSeg) }

    @run_decorator
    def run(self):

        # read the mc parameter
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        inp = self.input()
        rag = inp["Rag"].read()
        # FIXME why is this a vol target? it should easily fit in ram, or we are reaalllly screwed
        edgeCosts = inp["EdgeProbabilities"]
        edgeCosts.open()
        assert len(edgeCosts.shape) == 1
        edgeCosts = edgeCosts.read([0L],[long(edgeCosts.shape[0])])

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

        edgeCosts = weight_edge_costs(edgeCosts, rag, weighting_scheme, weight)

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
        save_path = os.path.join( PipelineParameter().cache, "StandardMulticutProblem.h5" )
        return HDF5DataTarget( save_path )


# modify the mc problem by changing the graph to the 'ModifiedAdjacency'
# and setting xy-edges that connect defected with non-defected nodes to be maximal repulsive
# TODO maybe weight down the skip edges by their respective skip - range
class ModifiedMulticutProblem(luigi.Task):

    pathToSeg = luigi.Parameter()
    # this can either contain a single path (classifier trained for xy - and z - edges jointly)
    # or two paths (classfier trained for xy - edges + classifier trained for z - edges separately)
    pathsToClassifier  = luigi.ListParameter()

    def requires(self):
        return {'edge_probabilities' : EdgeProbabilities(self.pathToSeg, self.pathsToClassifier),
                'modified_adjacency' : ModifiedAdjacency(self.pathToSeg),
                'rag' : StackedRegionAdjacencyGraph(self.pathToSeg)}

    @run_decorator
    def run(self):
        inp = self.input()
        modified_adjacency = inp['modified_adjacency']
        rag = inp['rag'].read()
        edge_costs = inp['edge_probabilities']
        edge_costs.open()
        edge_costs = edge_costs.read([0L],[long(edgeCosts.shape[0])])
        assert len(edge_costs.shape) == 1

        # read the mc parameter
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        # scale the probabilities
        # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
        p_min = 0.001
        p_max = 1. - p_min
        edge_costs = (p_max - p_min) * edge_costs + p_min

        beta = mc_config["beta"]

        # probabilities to energies, second term is boundary bias
        edge_costs = np.log( (1. - edge_costs) / edge_costs ) + np.log( (1. - beta) / beta )

        # weight edge costs
        weighting_scheme = mc_config["weightingScheme"]
        weight           = mc_config["weight"]

        edge_costs = weight_edge_costs(edge_costs, rag, weighting_scheme, weight)

        assert edge_costs.shape[0] == uvIds.shape[0]
        assert np.isfinite( edge_costs.min() ), str(edge_costs.min())
        assert np.isfinite( edge_costs.max() ), str(edge_costs.max())

        # modify the edges costs by setting the ignore edges to be maximally repulsive
        ignore_edges = modified_adjacency.read('ignore_edges')

        max_repulsive = 2 * edge_costs.max() # max is correct here !?!
        edge_costs[ignore_edges] = max_repulsive

        # TODO we might also want to weight down the skip-edges according to their range
        #skip_ranges = modified_adjacency.read('skip_ranges')
        #skip_edges_begin = rag.numberOfEdges
        #assert edge_costs.shape[0] - skip_edges_begin == len(skip_ranges), '%i, %i' % (edge_costs.shape[0] - skip_edges_begin, len(skip_ranges))
        #edge_costs[skip_edges_begin:] /= skip_ranges

        out = self.output()

        out.write(edge_costs,'costs')
        # write the modified graph
        out.write(modified_adjacency.read('modified_adjacency'))

    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "ModifiedMulticutProblem.h5" )
        return HDF5DataTarget( save_path )


# select defect handling tasks if pipelineParams.defectPipeline
if PipelineParameter().defectPipeline:
    MulticutProblem = ModifiedMulticutProblem
else:
    MulticutProblem = StandardMulticutProblem
