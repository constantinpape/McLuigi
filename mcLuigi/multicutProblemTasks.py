# Multicut Pipeline implemented with luigi
# Multicut Problem Tasks

import luigi

import os
import time

from dataTasks import StackedRegionAdjacencyGraph
from learningTasks import EdgeProbabilities
from customTargets import HDF5DataTarget

from pipelineParameter import PipelineParameter
from tools import config_logger

import logging
import json

import numpy as np
import nifty

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

# get weights and uvids of the MC problem
class StandardMulticutProblem(luigi.Task):

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
        save_path = os.path.join( PipelineParameter().cache, "StandardMulticutProblem.h5" )
        return HDF5DataTarget( save_path )
