from __future__ import division, print_function

# Multicut Pipeline implemented with luigi
# Multicut Problem Tasks

import luigi

import os

from .dataTasks import StackedRegionAdjacencyGraph
from .learningTasks import EdgeProbabilities
from .customTargets import HDF5DataTarget
from .defectHandlingTasks import ModifiedAdjacency, SkipEdgeLengths
from .pipelineParameter import PipelineParameter
from .tools import config_logger, run_decorator

import logging

import numpy as np

# import the proper nifty version
try:
    import nifty
except ImportError:
    try:
        import nifty_with_cplex as nifty
    except ImportError:
        import nifty_with_gurobi as nifty

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


# get weights and uvids of the MC problem
class MulticutProblem(luigi.Task):

    pathToSeg = luigi.Parameter()
    # this can either contain a single path (classifier trained for xy - and z - edges jointly)
    # or two paths (classfier trained for xy - edges + classifier trained for z - edges separately)
    pathsToClassifier  = luigi.Parameter()
    keyToSeg = luigi.Parameter(default='data')

    def requires(self):
        return_tasks = {"edge_probabilities": EdgeProbabilities(self.pathToSeg,
                                                                self.pathsToClassifier,
                                                                self.keyToSeg),
                        "rag": StackedRegionAdjacencyGraph(self.pathToSeg,
                                                           self.keyToSeg)}
        # TODO these should also take the key to seg !
        if PipelineParameter().defectPipeline:
            assert False, "Defect mode is currently not supported !"
            return_tasks['modified_adjacency'] = ModifiedAdjacency(self.pathToSeg)
            return_tasks['skip_edge_lengths']  = SkipEdgeLengths(self.pathToSeg)
        return return_tasks

    @run_decorator
    def run(self):

        inp = self.input()

        edge_costs = inp["edge_probabilities"].read()
        assert edge_costs.ndim == 1
        workflow_logger.info("MulticutProblem: loaded edge probs of len %i" % len(edge_costs))


        if PipelineParameter().defectPipeline:
            workflow_logger.info("MulticutProblem: computing MulticutProblem for defect correction pipeline.")
            if inp['modified_adjacency'].read('has_defects'):
                self._modified_multicut_proplem(edge_costs)
            else:
                self._standard_multicut_problem(edge_costs)

        else:
            workflow_logger.info("MulticutProblem: computing MulticutProblem for standard pipeline.")
            self._standard_multicut_problem(edge_costs)

    # TODO parallelise ?!
    def _probabilities_to_costs(self, edge_costs):

        inp = self.input()

        # scale the probabilities
        # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
        p_min = 0.001
        p_max = 1. - p_min
        edge_costs = (p_max - p_min) * edge_costs + p_min

        beta = PipelineParameter().multicutBeta

        # probabilities to energies, second term is boundary bias
        edge_costs = np.log((1. - edge_costs) / edge_costs) + np.log((1. - beta) / beta)
        workflow_logger.info(
            "MulticutProblem: cost statistics before weighting: mean: %f, std: %f, min: %f, max: %f" % (
                np.mean(edge_costs),
                np.std(edge_costs),
                edge_costs.min(),
                edge_costs.max()
            )
        )

        # weight edge costs
        weighting_scheme = PipelineParameter().multicutWeightingScheme
        weight           = PipelineParameter().multicutWeight

        edgeLens = inp['rag'].readKey('edgeLengths')

        if PipelineParameter().defectPipeline:
            if inp["modified_adjacency"].read("has_defects"):
                skipLens = inp["skip_edge_lengths"].read()
                delete_edges = inp["modified_adjacency"].read("delete_edges")
                edgeLens = np.delete(edgeLens, delete_edges)
                edgeLens = np.concatenate([edgeLens, skipLens])
                workflow_logger.info("MulticutProblem: removed delete edges and added skipLens to edgeLens")
        assert edgeLens.shape[0] == edge_costs.shape[0], str(edgeLens.shape[0]) + " , " + str(edge_costs.shape[0])

        if weighting_scheme == "z":
            workflow_logger.info("MulticutProblem: weighting edge costs with scheme z and weight " + str(weight))

            edgeTransition = inp['rag'].readKey('totalNumberOfInSliceEdges')

            z_max = float(np.max(edgeLens[edgeTransition:]))
            # we only weight the z edges !
            w = weight * edgeLens[edgeTransition:] / z_max
            edge_costs[edgeTransition:] = np.multiply(w, edge_costs[edgeTransition:])

        elif weighting_scheme == "xyz":
            workflow_logger.info("MulticutProblem: weighting edge costs with scheme xyz and weight " + str(weight))

            edgeTransition = inp['rag'].readKey('totalNumberOfInSliceEdges')

            z_max = float(np.max(edgeLens[edgeTransition:]))
            xy_max = float(np.max(edgeLens[:edgeTransition]))
            w_z = weight * edgeLens[edgeTransition:] / z_max
            w_xy = weight * edgeLens[:edgeTransition] / xy_max
            edge_costs[edgeTransition:] = np.multiply(w_z, edge_costs[edgeTransition:])
            edge_costs[:edgeTransition] = np.multiply(w_xy, edge_costs[:edgeTransition])

        elif weighting_scheme == "all":
            workflow_logger.info("MulticutProblem: weighting edge costs with scheme all and weight " + str(weight))

            edge_max  = float(np.max(edgeLens))
            w  = weight * edgeLens / edge_max
            edge_costs = np.multiply(w, edge_costs)

        else:
            workflow_logger.info("MulticutProblem: using non-weighted edge costs")

        if weighting_scheme in ("z", "xyz", "all"):
            workflow_logger.info(
                "MulticutProblem: cost statistics after weighting: mean: %f, std: %f, min: %f, max: %f" % (
                    np.mean(edge_costs),
                    np.std(edge_costs),
                    edge_costs.min(),
                    edge_costs.max()
                )
            )

        return edge_costs

    def _modified_multicut_proplem(self, edge_costs):

        inp = self.input()

        # get the plain graph for the multicut problem, modified for
        inp = self.input()
        modified_adjacency = inp['modified_adjacency']
        g = nifty.graph.UndirectedGraph()
        g.deserialize(modified_adjacency.read('modified_adjacency'))

        # transform edge costs to probabilities
        edge_costs = self._probabilities_to_costs(edge_costs)

        # modify the edges costs by setting the ignore edges to be maximally repulsive
        ignore_edges = modified_adjacency.read('ignore_edges')

        if ignore_edges.size:
            max_repulsive = 2 * edge_costs.min()  # TODO min correct here !?!
            edge_costs[ignore_edges] = max_repulsive

        # TODO we might also want to weight down the skip-edges according to their range
        # skip_ranges = modified_adjacency.read('skip_ranges')
        # skip_edges_begin = rag.numberOfEdges
        # assert edge_costs.shape[0] - skip_edges_begin == len(skip_ranges), '%i, %i' % (
        #     edge_costs.shape[0] - skip_edges_begin,
        #     len(skip_ranges)
        # )
        # edge_costs[skip_edges_begin:] /= skip_ranges

        assert edge_costs.shape[0] == g.numberOfEdges, "%i, %i" % (edge_costs.shape[0], g.numberOfEdges)
        assert np.isfinite(edge_costs.min()), str(edge_costs.min())
        assert np.isfinite(edge_costs.max()), str(edge_costs.max())

        out = self.output()
        out.write(edge_costs, 'costs')
        out.write(g.serialize(), 'graph')
        out.write(g.numberOfNodes, 'number_of_nodes')

    def _standard_multicut_problem(self, edge_costs):

        inp = self.input()

        # construct the plain graph for the multicut problem
        uv_ids = inp['rag'].readKey('uvIds')
        n_vars = uv_ids.max() + 1
        assert n_vars == inp['rag'].readKey('numberOfNodes')
        g = nifty.graph.UndirectedGraph(int(n_vars))
        g.insertEdges(uv_ids)

        # transform edge costs to probabilities
        edge_costs = self._probabilities_to_costs(edge_costs)

        assert edge_costs.shape[0] == uv_ids.shape[0]
        assert np.isfinite(edge_costs.min()), str(edge_costs.min())
        assert np.isfinite(edge_costs.max()), str(edge_costs.max())

        # write concatenation of uvids and edge costs
        out = self.output()

        assert g.numberOfEdges == edge_costs.shape[0]
        out.write(g.serialize(), "graph")
        out.write(edge_costs, "costs")
        out.write(g.numberOfNodes, 'number_of_nodes')

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "MulticutProblem_%s.h5" % (
                "modified" if PipelineParameter().defectPipeline else "standard",
            )
        )
        return HDF5DataTarget(save_path)
