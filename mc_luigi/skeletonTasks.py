# Multicut Pipeline implemented with luigi
# Workflow Tasks

import luigi
import logging
import numpy as np
from concurrent import futures
import os
import vigra

from pipelineParameter import PipelineParameter
from multicutProblemTasks import MulticutProblem
from blockwiseMulticutTasks import BlockwiseMulticutSolver
from nifty_helper import nifty_lmc_objective, run_nifty_lmc, nifty_lmc_fm_factory, nifty_lmc_kl_factory
from customTargets import HDF5DataTarget

from tools import config_logger, run_decorator, replace_from_dict

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

# import the proper nifty version
try:
    import nifty
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex as nifty
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        import nifty_with_gurobi as nifty
        import nifty_with_gurobi.graph.rag as nrag


# TODO criteria for the candidates
class CandidateObjects(luigi.Task):

    SegmentationPath = luigi.Parameter()

    def requires(self):
        pass

    # Find large candidadate objects in the segmentation and
    # store their bounding box and mask
    @run_decorator
    def run(self):
        pass

    def output(self):
        pass


class SkeletonsFromCandidates(luigi.Task):

    SegmentationPath = luigi.Parameter()

    def requires(self):
        pass

    # Skeletonize the candidate objects
    @run_decorator
    def run(self):
        pass

    def output(self):
        pass


# features for skeletons
# TODO figure out what we want here and if we actually treat this as features or directly as weights for the lmc
# potential features:
# -> nblast
# -> skeleton graph properties
# -> ???
class SkeletonFeatures(luigi.Task):

    # the skeletons, TODO wrap different skeletons via tasks (external skeletons, skeletons from candidates)
    Skeletons = luigi.TaskParameter()

    def requires(self):
        pass

    @run_decorator
    def run(self):
        pass

    def output(self):
        pass


# CandidateObjects, should be saved in the following format:
# "objects": Array with the ids of objects that should be resolved
# "lifted_uvs": Array with uv-ids corresponding to the lifted edges for each object
# "lifted_costs": Array wih the costs for the lifted edges for each object
class GenerateCandidateObjects(luigi.Task):

    #
    SegmentationPath = luigi.Parameter()

    def requires(self):
        pass

    @run_decorator
    def run(self):
        pass

    def output(self):
        pass


class ResolveCandidates(luigi.Task):

    segmentationPath = luigi.Parameter()
    fragmentationPath = luigi.Parameter()
    fragmentClassifierPath = luigi.Parameter()
    weight = luigi.Parameter(default=1.)
    numberOfLevels = luigi.Parameter(default=1.)

    def requires(self):
        mc_problem = MulticutProblem(self.fragmentationPath, self.fragmentClassifierPath)
        return {
            "mc_problem": mc_problem,
            "candidate_objects": GenerateCandidateObjects(self.segmentationPath),
            "mc_nodes": BlockwiseMulticutSolver(self.segmentationPath, mc_problem, self.numberOfLevels)
        }

    @run_decorator
    def run(self):

        inp = self.input()
        local_costs = inp["mc_problem"].read("costs")
        local_graph = inp["mc_problem"].read("graph")
        mc_nodes = inp["mc_nodes"].read()

        candidates = inp["candidate_objects"].read("objects")
        # TODO lifted uvs need to be reshaped properly!
        lifted_uvs_to_candidates = inp["candidate_objects"].read("lifted_uv_ids")
        lifted_costs_to_candidates = inp["candidate_objects"].read("lifted_costs")

        def resolve_candidate(i, object_id):

            # find the nodes in the fragmentation that belong to this object
            fragment_nodes = np.where(mc_nodes == object_id)[0]
            lifted_uv_ids = lifted_uvs_to_candidates[i]
            lifted_costs = lifted_costs_to_candidates[i]

            # get the local edge-ids
            inner_edges, outer_edges, subgraph = local_graph.extractSubgraphFromNodes(fragment_nodes)
            costs = local_costs[inner_edges]

            # this is how the global nodes are mapped to local nodes
            # in 'extractSubgraphFromNodes'
            # TODO would be nice to also get this from the nifty function directly
            global_to_local_nodes = {node: i for i, node in enumerate(fragment_nodes)}

            uv_ids = subgraph.uvIds()
            lifted_uv_ids = replace_from_dict(lifted_uv_ids, global_to_local_nodes)

            # properly weight the local and lifted edges
            costs /= len(uv_ids)
            lifted_costs /= len(lifted_uv_ids)
            lifted_costs *= self.weight

            # resolve the object with a lifted multicut
            lmc_objective = nifty_lmc_objective(
                uv_ids,
                lifted_uv_ids,
                costs,
                lifted_costs
            )

            return fragment_nodes, run_nifty_lmc(
                lmc_objective,
                nifty_lmc_fm_factory(
                    lmc_objective,
                    nifty_lmc_kl_factory(
                        lmc_objective,
                        warmstart=True
                    ),
                    warmstart=True
                )
            )

        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as tp:
            tasks = [tp.submit(resolve_candidate, i, object_id) for i, object_id in enumerate(candidates)]
            results = [t.result() for t in tasks]

        offset = mc_nodes.max() + 1
        resolved_node_result = mc_nodes.copy()
        for fragment_nodes, lmc_res in results:
            lmc_res += offset
            offset = lmc_res.max() + 1
            resolved_node_result[fragment_nodes] = lmc_res

        resolved_node_result = vigra.analysis.relabelConsecutive(
            resolved_node_result,
            start_label=0
        )
        self.output().write(resolved_node_result)

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "ResolvedSegmentation.h5"
        )
        return HDF5DataTarget(save_path)
