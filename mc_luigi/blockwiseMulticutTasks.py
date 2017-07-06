# Multicut Pipeline implemented with luigi
# Blockwise solver tasks

import luigi

from pipelineParameter import PipelineParameter
from dataTasks import ExternalSegmentation, StackedRegionAdjacencyGraph
from customTargets import HDF5DataTarget, FolderTarget
from defectDetectionTasks import DefectSliceDetection
from multicutProblemTasks import MulticutProblem
from blocking_helper import NodesToBlocks, EdgesBetweenBlocks, BlockGridGraph

from tools import config_logger, run_decorator, get_replace_slices, replace_from_dict
from nifty_helper import run_nifty_solver, string_to_factory, available_factorys

import os
import logging
import time
import h5py

import numpy as np
import vigra
from concurrent import futures

# import the proper nifty version
try:
    import nifty
    import nifty.graph.rag as nrag
    import nifty.hdf5 as nh5
    import nifty.ground_truth as ngt
except ImportError:
    try:
        import nifty_with_cplex as nifty
        import nifty_with_cplex.graph.rag as nrag
        import nifty_with_cplex.hdf5 as nh5
        import nifty_with_cplex.ground_truth as ngt
    except ImportError:
        import nifty_with_gurobi as nifty
        import nifty_with_gurobi.graph.rag as nrag
        import nifty_with_gurobi.hdf5 as nh5
        import nifty_with_gurobi.ground_truth as ngt

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


# parent class for blockwise solvers
class BlockwiseSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    globalProblem  = luigi.TaskParameter()
    numberOfLevels = luigi.Parameter()

    def requires(self):
        # block size in first hierarchy level
        initialBlockShape = PipelineParameter().multicutBlockShape
        # block overlap, for now same for each hierarchy lvl
        block_overlap = PipelineParameter().multicutBlockOverlap

        problems = [self.globalProblem]
        block_factor = 1

        for l in xrange(self.numberOfLevels):
            block_shape = map(lambda x: x * block_factor, initialBlockShape)

            # TODO check that we don't get larger than the actual shape here
            problems.append(
                ReducedProblem(self.pathToSeg, problems[-1], block_shape, block_overlap, l)
            )
            block_factor *= 2

        return problems

    def run(self):
        raise NotImplementedError("BlockwiseSolver is abstract and does not implement a run functionality!")

    # map back to the global solution
    def map_node_result_to_global(self, problems, reduced_node_result):

        n_nodes_global = problems[0].read('number_of_nodes')
        reduced_problem = problems[-1]
        to_global_nodes = reduced_problem.read("new2global")

        # TODO vectorize
        node_result = np.zeros(n_nodes_global, dtype='uint32')
        for node_id, node_res in enumerate(reduced_node_result):
            node_result[to_global_nodes[node_id]] = node_res

        return node_result

    def output(self):
        raise NotImplementedError("BlockwiseSolver is abstract and does not implement the output functionality!")


# TODO test !!!
# stitch blockwise sub-results by overlap
class BlockwiseOverlapSolver(BlockwiseSolver):

    def requires(self):

        # get the problem hierarchy from the super class
        problems = super(BlockwiseOverlapSolver, self).requires()

        # get the overlap
        overlap = PipelineParameter().multicutBlockOverlap

        # get the block shape of the current level
        initial_shape = PipelineParameter().multicutBlockShape
        block_factor  = max(1, (self.numberOfLevels - 1) * 2)
        block_shape = list(map(lambda x: x * block_factor, initial_shape))

        # get the sub solver results
        sub_solver = BlockwiseSubSolver(
            self.pathToSeg,
            problems[-2],
            block_shape,
            overlap,
            self.numberOfLevels - 1,
            True
        )

        return {
            'subblocks': SubblockSegmentations(self.pathToSeg, self.globalProblem, self.numberOfLevels),
            'problems': problems,
            'block_graph': BlockGridGraph(
                self.pathToSeg,
                block_shape,
                overlap
            ),
            'sub_solver': sub_solver
        }

    @run_decorator
    def run(self):

        # get all inputs
        inp = self.input()
        subblocks = inp['subblocks']
        problems = inp['problems']
        block_graph = nifty.graph.UndirectedGraph()
        block_graph.deserialize(inp['block_graph'].read())
        sub_solver = inp['sub_solver']

        # read the relevant problem, which is the second to last reduced problem ->
        # because we merge according to results of last BlockwiseSubSolver
        reduced_problem = problems[-2]
        reduced_graph = nifty.graph.UndirectedGraph()
        reduced_graph.deserialize(reduced_problem.read("graph"))
        reduced_costs = reduced_problem.read("costs")
        reduced_objective = nifty.graph.optimization.multicut.multicutObjective(reduced_graph, reduced_costs)

        # find the node overlaps
        t_ovlp = time.time()
        node_overlaps = self._find_node_overlaps(subblocks, block_graph)
        workflow_logger.info(
            "BlockwiseOverlapSolver: extracting overlapping nodes from blocks in %f s" % (time.time() - t_ovlp,)
        )

        # stitch the blocks based on the node overlaps
        t_stitch = time.time()
        reduced_node_result = self._stitch_blocks(block_graph, node_overlaps, reduced_graph, sub_solver, reduced_graph)
        workflow_logger.info(
            "BlockwiseOverlapSolver: stitching blocks in %f s" % (time.time() - t_stitch,)
        )

        # get the energy of the solution
        workflow_logger.info(
            "BlockwiseOverlapSolver: Problem solved with energy %f"
            % reduced_objective.evalNodeLabels(reduced_node_result)
        )

        # map back to the global nodes and write result
        node_result = self.map_node_result_to_global(problems, reduced_node_result)
        self.output().write(node_result)

    # TODO test !!!
    def _find_node_overlaps(self, subblocks, block_graph):

        block_res_path = subblocks.path
        # read the nodes from the sub solver to convert back to
        # the actual problem node ids

        # extract the overlaps for all the edges
        def node_overlaps_for_block_pair(block_edge_id, block_uv):

            block_u, block_v = block_uv
            # get the uv-ids connecting the two blocks and the paths to the block segmentations
            block_u_path = os.path.join(block_res_path, 'block%i_segmentation.h5' % block_u)
            block_v_path = os.path.join(block_res_path, 'block%i_segmentation.h5' % block_v)

            # find the actual overlapping regions in block u and v and load them
            # TODO TODO TODO
            overlap_bb_u = 1
            overlap_bb_v = 2

            with h5py.File(block_u_path) as f_u, \
                    h5py.File(block_v_path) as f_v:

                seg_u = f_u['data'][overlap_bb_u]
                seg_v = f_v['data'][overlap_bb_v]

            n_nodes_u = np.max(seg_u) + 1
            # find the overlaps between the two segmentations
            overlap_counter = ngt.Overlap(n_nodes_u - 1, seg_u, seg_v)

            return [overlap_counter.overlapArraysNormalized(node_u) for node_u in xrange(n_nodes_u)]

        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as tp:
            tasks = [
                tp.submit(node_overlaps_for_block_pair, block_edge_id, block_uv)
                for block_edge_id, block_uv in enumerate(block_graph.uvIds())
            ]
            # TODO merge the node overlaps ?!
            node_overlaps = [t.result() for t in tasks]

        return node_overlaps

    # TODO test !!!
    # TODO this is the most naive stitching for now (always stitch max overlap).
    # once this works, try better stitching heuristics
    def _stitch_blocks(self, block_graph, node_overlaps, sub_solver, reduced_graph):

        n_blocks = block_graph.nuberOfNodes
        # read the sub_node results
        sub_node_results = sub_solver.read('sub_results')
        assert len(sub_node_results) == n_blocks

        # apply offset to each sub-block result to have unique ids before stitching
        offset = 0
        block_offsets = []
        for sub_res in sub_node_results:
            block_offsets.append(offset)
            sub_res += offset
            offset += sub_res.max() + 1

        # create ufd to merge with last offset value -> number of nodes that need to be merged
        ufd = nifty.ufd.ufd(offset)

        # now, we iterate over the block pairs and merge nodes according to their overlap
        for block_pair_id, block_u, block_v in enumerate(block_graph.uvIds()):

            # get the results from the overlap calculations
            overlapping_nodes, overlaps = node_overlaps[block_pair_id]

            offsets_u = block_offsets[block_u]
            offsets_v = block_offsets[block_v]

            # iterate over the nodes in overlap(u) and merge with nodes in overlap(v)
            # according to the overlaps
            # TODO for now we simply merge the maximum overlap node,
            # but we want to do more clever things eventually
            for node_u_id, nodes_v in enumerate(overlapping_nodes):
                merge_node_v = nodes_v[0]  # TODO is this ordered in descending order by nifty ?
                ufd.merge(node_u_id + offsets_u, node_v + offsets_v)

        # get the merge result
        node_result = ufd.elementLabeling()

        # project back to the reduced problem nodes via iterating over the blocks and projection of
        # the block node result
        reduced_node_result = np.zeros(reduced_graph.numberOfNodes, dtyp='uint32')
        sub_nodes = sub_solver.read('sub_nodes')

        for block_id in xrange(n_blocks):

            # first find the result for the nodes in this block
            block_result = node_result[block_offsets[block_id], block_offsets[block_id + 1]] \
                if block_id < n_blocks -1 else node_result[block_offsets[block_id]:offset]

            # next, map the merge result to the sub-result nodes
            sub_result = sub_node_results[block_id]
            reduced_result = block_result[sub_result]

            # finally, write to the block result
            reduced_node_result[sub_nodes[block_id]] = reduced_result

        return reduced_node_result

    def output(self):
        save_name = "BlockwiseOverlapSolver_L%i_%s_%s_%s.h5" % (
            self.numberOfLevels,
            '_'.join(map(str, PipelineParameter().multicutBlockShape)),
            '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
            "modified" if PipelineParameter().defectPipeline else "standard"
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


# Produce the sub-block segmentations for debugging
class SubblockSegmentations(BlockwiseSolver):

    def requires(self):
        problems = super(SubblockSegmentations, self).requires()
        block_shape = []

        # block size in first hierarchy level
        block_factor = (self.numberOfLevels - 1) * 2 if self.numberOfLevels > 1 else 1
        block_shape = map(
            lambda x: x * block_factor,
            PipelineParameter().multicutBlockShape
        )
        block_overlap = PipelineParameter().multicutBlockOverlap

        sub_solver = BlockwiseSubSolver(
            self.pathToSeg,
            problems[-2],
            block_shape,
            block_overlap,
            self.numberOfLevels - 1,
            True
        )
        return_tasks = {
            'sub_solver': sub_solver,
            'rag': StackedRegionAdjacencyGraph(self.pathToSeg)
        }

        if PipelineParameter().defectPipeline:
            return_tasks["defect_slices"] = DefectSliceDetection(self.pathToSeg)

        return return_tasks

    @run_decorator
    def run(self):

        # read stuff from the sub solver
        sub_solver = self.input()['sub_solver']
        sub_results = sub_solver.read('sub_results')
        block_begins = sub_solver.read('block_begins')
        block_ends = sub_solver.read('block_ends')
        sub_nodes = sub_solver.read('sub_nodes')

        has_defects = False
        if PipelineParameter().defectPipeline:
            defect_slices_path = self.input()['defect_slices'].path
            defect_slices = vigra.readHDF5(defect_slices_path, 'defect_slices')
            if defect_slices.size:
                has_defects = True

        # get the rag
        rag = self.input()['rag'].read()

        out_path = self.output().path
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        # iterate over the blocks and serialize the sub-block result
        # for block_id in xrange(1):
        for block_id in xrange(len(sub_results)):
            sub_result = {sub_nodes[block_id][i]: sub_results[block_id][i]
                          for i in xrange(len(sub_nodes[block_id]))}

            print "Saving Block-Result for block %i / %i" % (block_id, len(sub_results))
            block_begin = block_begins[block_id]
            block_end = block_ends[block_id]

            # save the begin and end coordinates of this block for later use
            block_path = os.path.join(out_path, 'block%i_coordinates.h5' % block_id)
            vigra.writeHDF5(block_begin, block_path, 'block_begin')
            vigra.writeHDF5(block_end, block_path, 'block_end')

            # determine the shape of this subblock
            block_shape = block_end - block_begin
            chunk_shape = [1, min(512, block_shape[1]), min(512, block_shape[2])]

            # save the segmentation for this subblock
            res_path = os.path.join(out_path, 'block%i_segmentation.h5' % block_id)
            res_file = nh5.createFile(res_path)
            out_array = nh5.Hdf5ArrayUInt32(
                res_file,
                'data',
                block_shape.tolist(),
                chunk_shape,
                compression=PipelineParameter().compressionLevel
            )

            nrag.projectScalarNodeDataInSubBlock(
                rag,
                sub_result,
                out_array,
                map(long, block_begins[block_id]),
                map(long, block_ends[block_id])
            )

            # if we have defected slices in this sub-block, replace them by an adjacent slice
            if has_defects:

                # project the defected slicces in global coordinates to the subblock coordinates
                this_defect_slices = defect_slices - block_begin[0]
                this_defect_slices = this_defect_slices[
                    np.logical_and(this_defect_slices > 0, this_defect_slices < block_shape[0])
                ]

                # only replace slices if there are any in the subblock
                if this_defect_slices.size:
                    replace_slice = get_replace_slices(this_defect_slices, block_shape)
                    for z in this_defect_slices:
                        replace_z = replace_slice[z]
                        workflow_logger.debug(
                            "SubblockSegmentationWorkflow: block %i replacing defected slice %i by %i"
                            % (block_id, z, replace_z)
                        )
                        out_array.writeSubarray(
                            [z, 0L, 0L],
                            out_array.readSubarray([replace_z, 0L, 0L], [replace_z + 1, block_shape[1], block_shape[2]])
                        )

            nh5.closeFile(res_file)

    def output(self):
        # we add the number of levels, the initial block shape and the block overlap
        # to the save name to make int unambiguous
        save_name = "SubblockSegmentations_L%i_%s_%s_%s" % (
            self.numberOfLevels,
            '_'.join(map(str, PipelineParameter().multicutBlockShape)),
            '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
            "modified" if PipelineParameter().defectPipeline else "standard"
        )
        return FolderTarget(
            os.path.join(PipelineParameter().cache, save_name)
        )


# produce reduced global graph from subproblems
# solve global multicut problem on the reduced graph
class BlockwiseMulticutSolver(BlockwiseSolver):

    @run_decorator
    def run(self):

        problems = self.input()

        # we solve the problem for the costs and the edges of the last level of hierarchy
        reduced_problem = problems[-1]

        reduced_graph = nifty.graph.UndirectedGraph()
        reduced_graph.deserialize(reduced_problem.read("graph"))
        reduced_costs = reduced_problem.read("costs")
        reduced_objective = nifty.graph.optimization.multicut.multicutObjective(reduced_graph, reduced_costs)

        #
        # run global multicut inference
        #

        # we use fm with kl as default backend, because this shows the best scaling behaviour
        solver_type = PipelineParameter().globalSolverType
        inf_params  = dict(
            sigma=PipelineParameter().multicutSigmaFusion,
            number_of_iterations=PipelineParameter().multicutNumIt,
            n_stop=PipelineParameter().multicutNumItStopGlobal,
            n_threads=PipelineParameter().multicutNThreadsGlobal,
            n_fuse=PipelineParameter().multicutNumFuse,
            seed_fraction=PipelineParameter().multicutSeedFractionGlobal
        )

        workflow_logger.info("BlockwiseMulticutSolver: Solving problems with solver %s" % solver_type)
        workflow_logger.info(
            "BlockwiseMulticutSolver: With Params %s" % ' '.join(
                ['%s, %s,' % (str(k), str(v)) for k, v in inf_params.iteritems()]
            )
        )

        factory = string_to_factory(reduced_objective, solver_type, inf_params)
        reduced_node_result, energy, t_inf = run_nifty_solver(
            reduced_objective,
            factory,
            verbose=True,
            time_limit=PipelineParameter().multicutGlobalTimeLimit
        )
        workflow_logger.info(
            "BlockwiseMulticutSolver: Inference of reduced problem for the whole volume took: %f s" % t_inf
        )

        # NOTE: we don't need to project back to global problem to calculate the correct energy !
        workflow_logger.info(
            "BlockwiseMulticutSolver: Problem solved with energy %f"
            % reduced_objective.evalNodeLabels(reduced_node_result)
        )

        node_result = self.map_node_result_to_global(problems, reduced_node_result)
        self.output().write(node_result)

    def output(self):
        save_name = "BlockwiseMulticutSolver_L%i_%s_%s_%s.h5" % (
            self.numberOfLevels,
            '_'.join(map(str, PipelineParameter().multicutBlockShape)),
            '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
            "modified" if PipelineParameter().defectPipeline else "standard"
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


# TODO debug !!!
# stitch blockwise sub-results according to costs of the edges connecting the sub-blocks
class BlockwiseStitchingSolver(BlockwiseSolver):

    boundaryBias = luigi.Parameter(default=.95)

    @run_decorator
    def run(self):
        problems = self.input()
        reduced_problem = problems[-1]

        # load the reduced graph of the current level
        reduced_graph = nifty.graph.UndirectedGraph()
        reduced_graph.deserialize(reduced_problem.read("graph"))
        reduced_costs = reduced_problem.read("costs")
        reduced_objective = nifty.graph.optimization.multicut.multicutObjective(reduced_graph, reduced_costs)

        uv_ids = reduced_graph.uvIds()
        outer_edges = reduced_problem.read('outer_edges')

        workflow_logger.info(
            "BlockwiseStitchingSolver: Looking for merge edges in %i between block edges of %i total edges"
            % (len(outer_edges), len(uv_ids))
        )

        # merge all edges along the block boundaries that are attractive
        energyBias = 0 if self.boundaryBias == .5 else \
            np.log((1. - self.boundaryBias) / self.boundaryBias)
        merge_ids = outer_edges[reduced_costs[outer_edges] < energyBias]

        workflow_logger.info(
            "BlockwiseStitchingSolver: Merging %i edges with value smaller than bias %f of %i between block edges"
            % (len(merge_ids), energyBias, len(outer_edges))
        )

        ufd = nifty.ufd.ufd(reduced_graph.numberOfNodes)
        ufd.merge(uv_ids[merge_ids])
        reduced_node_result = ufd.elementLabeling()

        workflow_logger.info(
            "BlockwiseStitchingSolver: Problem solved with energy %f"
            % reduced_objective.evalNodeLabels(reduced_node_result)
        )

        node_result = self.map_node_result_to_global(problems, reduced_node_result)
        self.output().write(node_result)

    def output(self):
        save_name = "BlockwiseStitchingSolver_L%i_%s_%s_%s.h5" % (
            self.numberOfLevels,
            '_'.join(map(str, PipelineParameter().multicutBlockShape)),
            '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
            "modified" if PipelineParameter().defectPipeline else "standard"
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


# stitch blockwise sub-results by running Multicut only on the edges between blocks
# -> no idea if this will actually work
# -> two options:
# --> run Multicut on all between block edges
# --> run Multicuts for all pairs of adjacent blocks and check edges that are part of multiple
#     block pairs for consistency -> if inconsistent, don't merge
# -> maybe this needs a different problem formulation than Multicut ?!

# TODO implement
def BlockwiseMulticutStitchingSolver(BlockwiseSolver):

    # we have EdgesBetweenBlocks as additional requirements to the
    # super class (BlockwiseSolver)
    def requires(self):
        problems = super(BlockwiseMulticutStitchingSolver, self).requires()
        overlap = PipelineParameter().multicutBlockOverlap

        # get the block shape of the current level
        initial_shape = PipelineParameter().multicutBlockShape
        block_factor  = max(1, (self.numberOfLevels - 1) * 2)
        block_shape = list(map(lambda x: x * block_factor, initial_shape))

        return {
            'problems': problems,
            'edges_between_blocks': EdgesBetweenBlocks(
                self.pathToSeg,
                problems[-1],
                block_shape,
                overlap,
                self.numberOfLevels
            )
        }

    @run_decorator
    def run(self):
        problems = self.input()['problems']
        reduced_problem = problems[-1]

        # load the reduced graph of the current level
        reduced_graph = nifty.graph.UndirectedGraph()
        reduced_graph.deserialize(reduced_problem.read("graph"))
        reduced_costs = reduced_problem.read("costs")
        reduced_objective = nifty.graph.optimization.multicut.multicutObjective(reduced_graph, reduced_costs)

        t_extract = time.time()
        sub_problems = self._extract_subproblems(reduced_graph, reduced_costs)
        workflow_logger.info("BlockwiseMulticutStitchingSolver: Problem extraction took %f s" % (time.time() - t_extract))

        t_solve = time.time()
        edge_results = self._extract_subproblems(sub_problems, reduced_costs, reduced_graph.numberOfEdges)
        workflow_logger.info("BlockwiseMulticutStitchingSolver: Problem solving took %f s" % (time.time() - t_solve))

        t_merge = time.time()
        reduced_node_result = self._merge_blocks(reduced_graph, sub_problems, edge_result)
        workflow_logger.info("BlockwiseMulticutStitchingSolver: Problem solving took %f s" % (time.time() - t_solve))

        workflow_logger.info(
            "BlockwiseMulticutStitchingSolver: Problem solved with energy %f"
            % reduced_objective.evalNodeLabels(reduced_node_result)
        )

        node_result = self.map_node_result_to_global(problems, reduced_node_result)
        self.output().write(node_result)

    def _extract_subproblems(self, reduced_graph):

        block_edges = self.input('edges_between_blocks')
        edges_between_blocks = block_edges.read('edges_between_blocks')
        uv_ids = reduced_graph.uvIds()

        def extract_subproblem(block_edge_id):
            this_edges = edges_between_blocks[block_edge_id]
            this_uvs = uv_ids[this_edges]
            this_nodes = np.unique(this_uvs)
            to_local_nodes = {node: i for i, node in enumerate(this_nodes)}
            g = nifty.UndirectedGraph(len(this_nodes))
            g.insertEdges(replace_from_dict(this_uvs, to_local_nodes))
            return g, this_edges

        return [extract_subproblem(block_edge_id) for block_edge_id in xrange(len(edges_between_blocks))]

    def _solve_subproblems(self, sub_problems, reduced_costs, number_of_edges):

        # we use the same sub-solver and settings as 'BlockwiseSubSolver'
        sub_solver_type = PipelineParameter().subSolverType
        if sub_solver_type in ('fm-ilp', 'fm-kl'):
            solver_params  = dict(
                sigma=PipelineParameter().multicutSigmaFusion,
                number_of_iterations=PipelineParameter().multicutNumIt,
                n_stop=PipelineParameter().multicutNumItStopGlobal,
                n_threads=0,
                n_fuse=PipelineParameter().multicutNumFuse,
                seed_fraction=PipelineParameter().multicutSeedFraction
            )
        else:
            solver_params = dict()

        workflow_logger.info("BlockwiseSubSolver: Solving sub-problems with solver %s" % sub_solver_type)
        workflow_logger.info(
            "BlockwiseSubSolver: With Params %s" % ' '.join(
                ['%s, %s,' % (str(k), str(v)) for k, v in solver_params.iteritems()]
            )
        )

        def mc(graph, costs):
            obj = nifty.graph.optimization.multicut.multicutObjective(g, costs)
            factory = string_to_factory(obj, sub_solver_type, solver_params)
            solver = factory.create(obj)
            return solver.optimize()

        # solve subproblems in parallel
        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as tp:
            tasks = [tp.submit(mc, prob[0], reduced_costs[prob[1]]) for prob in sub_problems]
            sub_results = [t.result() for t in tasks]

        edge_result = np.zeros(number_of_edges, dtype='uint8')

        # combine the subproblem results into global edge vector
        for problem_id in xrange(len(sub_problems)):
            node_result = sub_results[problem_id]
            sub_uv_ids = sub_problems[problem_id][0].uvIds()

            edge_sub_result = node_result[sub_uv_ids[:, 0]] != node_result[sub_uv_ids[:, 1]]

            edge_result[sub_problems[problem_id][1]] += edge_sub_result

        return edge_result

    def _merge_blocks(self, reduced_graph, edge_result):

        ufd = nifty.ufd.ufd(reduced_graph.numberOfNodes)
        uv_ids = reduced_graph.uvIds()

        merge_edges = uv_ids[edge_result == 0]
        ufd.merge(merge_edges)
        return ufd.elementLabeling()

    def output(self):
        save_name = "BlockwiseMulticutStitchingSolver_L%i_%s_%s_%s.h5" % (
            self.numberOfLevels,
            '_'.join(map(str, PipelineParameter().multicutBlockShape)),
            '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
            "modified" if PipelineParameter().defectPipeline else "standard"
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


class ReducedProblem(luigi.Task):

    pathToSeg = luigi.Parameter()
    problem   = luigi.TaskParameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()

    level = luigi.Parameter()

    def requires(self):
        return {
            "sub_solution": BlockwiseSubSolver(
                self.pathToSeg, self.problem, self.blockShape, self.blockOverlap, self.level
            ),
            "problem": self.problem
        }

    # TODO we need to recover the edges between blocks for the stitching solver
    @run_decorator
    def run(self):

        workflow_logger.info(
            "ReducedProblem: Reduced problem for level %i with block shape: %s"
            % (self.level, str(self.blockShape))
        )

        inp = self.input()
        problem   = inp["problem"]
        cut_edges = inp["sub_solution"].read()

        g = nifty.graph.UndirectedGraph()
        g.deserialize(problem.read("graph"))

        # merge the edges that are not cuts
        # and find the new nodes as well as the mapping
        # from new 2 old nodes

        # we time all the stuff for benchmarking
        t_merge = time.time()
        uv_ids = g.uvIds()

        ufd = nifty.ufd.ufd(g.numberOfNodes)

        merge_nodes = uv_ids[cut_edges == 0]
        ufd.merge(merge_nodes)

        old2new_nodes = ufd.elementLabeling()
        new2old_nodes = ufd.representativesToSets()
        # number of nodes for the new problem
        number_of_new_nodes = len(new2old_nodes)
        workflow_logger.info("ReducedProblem: Merging nodes took: %f s" % (time.time() - t_merge))

        t_edges = time.time()
        # find new edges and costs
        uv_ids_new = self.find_new_edges_and_costs(
            uv_ids,
            problem,
            cut_edges,
            number_of_new_nodes,
            old2new_nodes
        )
        workflow_logger.info("ReducedProblem: Computing new edges took: %f s" % (time.time() - t_edges))

        # serialize the node converter
        t_serialize = time.time()
        self.serialize_node_conversions(problem, old2new_nodes, new2old_nodes, number_of_new_nodes)
        workflow_logger.info("ReducedProblem: Serializing node converters took: %f s" % (time.time() - t_serialize))

        workflow_logger.info("ReucedProblem: Nodes: From %i to %i" % (g.numberOfNodes, number_of_new_nodes))
        workflow_logger.info("ReucedProblem: Edges: From %i to %i" % (g.numberOfEdges, len(uv_ids_new)))

    def find_new_edges_and_costs(
        self,
        uv_ids,
        problem,
        cut_edges,
        number_of_new_nodes,
        old2new_nodes
    ):

        # find mapping from new to old edges with nifty impl
        edge_mapping = nifty.tools.EdgeMapping(len(uv_ids))
        edge_mapping.initializeMapping(uv_ids, old2new_nodes)

        # get the new uv-ids
        uv_ids_new = edge_mapping.getNewUvIds()

        # map the old costs to new costs
        costs  = problem.read("costs")
        new_costs = edge_mapping.mapEdgeValues(costs)

        # map the old outer edges to new outer edges
        outer_edge_ids = self.input()["sub_solution"].read("outer_edges")
	new_outer_edges = edge_mapping.getNewEdgeIds(outer_edge_ids)

        assert len(new_costs) == len(uv_ids_new)
        reduced_graph = nifty.graph.UndirectedGraph(number_of_new_nodes)
        reduced_graph.insertEdges(uv_ids_new)

        out = self.output()
        out.write(reduced_graph.serialize(), "graph")
        out.write(reduced_graph.numberOfNodes, 'number_of_nodes')
        out.write(new_costs, "costs")
	out.write(new_outer_edges, 'outer_edges')

        return uv_ids_new

    def serialize_node_conversions(self, problem, old2new_nodes, new2old_nodes, number_of_new_nodes):

        if self.level == 0:
            global2new = old2new_nodes
            new2global = new2old_nodes

        else:
            global2new_last = problem.read("global2new").astype(np.uint32)
            new2global_last = problem.read("new2global")

            global2new = -1 * np.ones_like(global2new_last, dtype=np.int32)
            new2global = []

            # TODO vectorize
            for newNode in xrange(number_of_new_nodes):
                oldNodes = new2old_nodes[newNode]
                globalNodes = np.concatenate(new2global_last[oldNodes])
                global2new[globalNodes] = newNode
                new2global.append(globalNodes)

        assert -1 not in global2new
        global2new = global2new.astype('uint32')

        out = self.output()
        out.write(global2new, "global2new")

        # need to serialize this differently, due to list of list
        new2old_nodes = np.array([np.array(n2o, dtype='uint32') for n2o in new2old_nodes])
        out.writeVlen(new2old_nodes, "new2old")

        new2global = np.array([np.array(n2g, dtype='uint32') for n2g in new2global])
        out.writeVlen(new2global, "new2global")

    def output(self):
        save_name = "ReducedProblem_L%i_%s_%s_%s.h5" % (
            self.level,
            '_'.join(map(str, self.blockShape)),
            '_'.join(map(str, self.blockOverlap)),
            "modified" if PipelineParameter().defectPipeline else "standard"
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


class BlockwiseSubSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    problem   = luigi.TaskParameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()

    level = luigi.Parameter()
    # needs to be true if we want to use the stitching - by overlap solver
    serializeSubResults = luigi.Parameter(default=False)

    def requires(self):
        initialShape = PipelineParameter().multicutBlockShape
        overlap      = PipelineParameter().multicutBlockOverlap

        nodes2blocks = NodesToBlocks(self.pathToSeg, initialShape, overlap)
        return {"seg": ExternalSegmentation(self.pathToSeg), "problem": self.problem, "nodes2blocks": nodes2blocks}

    @run_decorator
    def run(self):
        # Input
        inp = self.input()
        seg = inp["seg"]
        seg.open()
        problem = inp["problem"]
        costs = problem.read("costs")
        nodes2blocks = inp["nodes2blocks"].read()

        graph = nifty.graph.UndirectedGraph()
        graph.deserialize(problem.read("graph"))
        number_of_edges = graph.numberOfEdges

        global2new_nodes = None if self.level == 0 else problem.read("global2new")

        workflow_logger.info("BlockwiseSubSolver: Starting extraction of subproblems.")
        t_extract = time.time()
        subproblems = self._run_subproblems_extraction(seg, graph, nodes2blocks, global2new_nodes)
        workflow_logger.info("BlockwiseSubSolver: Extraction time for subproblems %f s" % (time.time() - t_extract,))

        workflow_logger.info("BlockwiseSubSolver: Starting solvers for subproblems.")
        t_inf_total = time.time()
        self._solve_subproblems(costs, subproblems, number_of_edges)
        workflow_logger.info(
            "BlockwiseSubSolver: Inference time total for subproblems %f s" % (time.time() - t_inf_total,)
        )

        seg.close()

    # extract all sub-problems for current level
    def _run_subproblems_extraction(self, seg, graph, nodes2blocks, global2new_nodes):

        # get the initial blocking
        # block size in first hierarchy level
        initial_block_shape = PipelineParameter().multicutBlockShape
        initial_overlap = list(PipelineParameter().multicutBlockOverlap)
        initial_blocking = nifty.tools.blocking(
            roiBegin=[0L, 0L, 0L],
            roiEnd=seg.shape(),
            blockShape=initial_block_shape
        )
        workflow_logger.info(
            "BlockwiseSubSolver: Extracting sub-problems with initial blocking of shape %s with overlaps %s."
            % (str(initial_block_shape), str(initial_overlap))
        )

        # function for subproblem extraction
        # extraction only for level 0
        def extract_subproblem(block_id, sub_blocks):
            node_list = np.unique(np.concatenate([nodes2blocks[sub_id] for sub_id in sub_blocks]))
            if self.level != 0:
                node_list = np.unique(global2new_nodes[node_list])
            workflow_logger.debug(
                "BlockwiseSubSolver: block id %i: Number of nodes %i" % (block_id, node_list.shape[0])
            )
            inner_edges, outer_edges, subgraph = graph.extractSubgraphFromNodes(node_list.tolist())
            return np.array(inner_edges), np.array(outer_edges), subgraph, node_list

        block_overlap = list(self.blockOverlap)
        blocking = nifty.tools.blocking(roiBegin=[0L, 0L, 0L], roiEnd=seg.shape(), blockShape=self.blockShape)
        number_of_blocks = blocking.numberOfBlocks

        workflow_logger.info(
            "BlockwiseSubSolver: Extracting sub-problems with current blocking of shape %s with overlaps %s."
            % (str(self.blockShape), str(block_overlap))
        )

        n_workers = min(number_of_blocks, PipelineParameter().nThreads)
        # parallel
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = []
            for block_id in xrange(number_of_blocks):

                # get the current block with additional overlap
                block = blocking.getBlockWithHalo(block_id, block_overlap).outerBlock
                block_begin, block_end = block.begin, block.end
                workflow_logger.debug(
                    "BlockwiseSubSolver: block id %i start %s end %s" % (block_id, str(block_begin), str(block_end))
                )

                # if we are past level 0, we must assemble the initial blocks, from which this block is made up of
                # otherwise we simply schedule this block
                if self.level > 0:
                    sub_blocks = initial_blocking.getBlockIdsInBoundingBox(block_begin, block_end, initial_overlap)
                else:
                    sub_blocks = [block_id]

                tasks.append(executor.submit(extract_subproblem, block_id, sub_blocks))
            sub_problems = [task.result() for task in tasks]

        out = self.output()
        # we need to serialize the outer edges for the stitching solvers
        out.write(
            np.unique(np.concatenate([sub_prob[1] for sub_prob in sub_problems])),
            'outer_edges'
        )

        # if we serialize the sub-results, write out the block positions and the sub nodes here
        if self.serializeSubResults:
            out.write(
                np.concatenate([
                    np.array(blocking.getBlockWithHalo(block_id, block_overlap).outerBlock.begin)[None, :]
                    for block_id in xrange(number_of_blocks)],
                    axis=0
                ),
                'block_begins'
            )
            out.write(
                np.concatenate([
                    np.array(blocking.getBlockWithHalo(block_id, block_overlap).outerBlock.end)[None, :]
                    for block_id in xrange(number_of_blocks)],
                    axis=0
                ),
                'block_ends'
            )
            out.writeVlen(
                np.array([sub_prob[3] for sub_prob in sub_problems]),
                'sub_nodes'
            )

        assert len(sub_problems) == number_of_blocks, str(len(sub_problems)) + " , " + str(number_of_blocks)
        return sub_problems

    def _solve_subproblems(self, costs, sub_problems, number_of_edges):

        sub_solver_type = PipelineParameter().subSolverType
        if sub_solver_type in ('fm-ilp', 'fm-kl'):
            solver_params  = dict(
                sigma=PipelineParameter().multicutSigmaFusion,
                number_of_iterations=PipelineParameter().multicutNumIt,
                n_stop=PipelineParameter().multicutNumItStopGlobal,
                n_threads=0,
                n_fuse=PipelineParameter().multicutNumFuse,
                seed_fraction=PipelineParameter().multicutSeedFraction
            )
        else:
            solver_params = dict()

        workflow_logger.info("BlockwiseSubSolver: Solving sub-problems with solver %s" % sub_solver_type)
        workflow_logger.info(
            "BlockwiseSubSolver: With Params %s" % ' '.join(
                ['%s, %s,' % (str(k), str(v)) for k, v in solver_params.iteritems()]
            )
        )

        def _solve_mc(g, costs, block_id):
            workflow_logger.debug(
                "BlockwiseSubSolver: Solving MC Problem for block %i with %i nodes / %i edges"
                % (block_id, g.numberOfNodes, g.numberOfEdges)
            )
            obj = nifty.graph.optimization.multicut.multicutObjective(g, costs)
            factory = string_to_factory(obj, sub_solver_type, solver_params)
            solver = factory.create(obj)
            t_inf  = time.time()
            res    = solver.optimize()
            workflow_logger.debug(
                "BlockwiseSubSolver: Inference for block %i with fusion moves solver in %f s"
                % (block_id, time.time() - t_inf)
            )
            return res

        # sequential for debugging
        # subResults = []
        # for blockId, subProblem in enumerate(sub_problems):
        #     print "Sequential prediction for block id:", blockId
        #     subResults.append( _solve_mc( subProblem[2], costs[subProblem[0]], blockId) )

        n_workers = min(len(sub_problems), PipelineParameter().nThreads)
        # n_workers = 1
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(
                _solve_mc,
                sub_problem[2],
                costs[sub_problem[0]],
                block_id) for block_id, sub_problem in enumerate(sub_problems)]
        sub_results = [task.result() for task in tasks]

        cut_edges = np.zeros(number_of_edges, dtype=np.uint8)

        assert len(sub_results) == len(sub_problems), str(len(sub_results)) + " , " + str(len(sub_problems))

        for block_id in xrange(len(sub_problems)):

            # get the cut edges from the subproblem
            node_result = sub_results[block_id]
            sub_uv_ids = sub_problems[block_id][2].uvIds()

            edge_result = node_result[sub_uv_ids[:, 0]] != node_result[sub_uv_ids[:, 1]]

            cut_edges[sub_problems[block_id][0]] += edge_result
            # add up outer edges
            cut_edges[sub_problems[block_id][1]] += 1

        # all edges which are cut at least once will be cut
        out = self.output()
        cut_edges[cut_edges >= 1] = 1
        out.write(cut_edges)

        # if we serialize the sub results, write them here
        if self.serializeSubResults:
            out.writeVlen(
                np.array([sub_res for sub_res in sub_results]),
                'sub_results'
            )

    def output(self):
        save_name = "BlockwiseSubSolver_L%i_%s_%s_%s.h5" % (
            self.level,
            '_'.join(map(str, self.blockShape)),
            '_'.join(map(str, self.blockOverlap)),
            "modified" if PipelineParameter().defectPipeline else "standard"
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


# only works for level 1 for now!
class TestSubSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToClassifier = luigi.Parameter()

    blockShape   = luigi.ListParameter(default=PipelineParameter().multicutBlockShape)
    blockOverlap = luigi.ListParameter(default=PipelineParameter().multicutBlockOverlap)

    def requires(self):
        nodes2blocks = NodesToBlocks(self.pathToSeg, self.blockShape, self.blockOverlap)
        return {
            "seg": ExternalSegmentation(self.pathToSeg),
            "problem": MulticutProblem(self.pathToSeg, self.pathToClassifier),
            "nodes2blocks": nodes2blocks
        }

    @run_decorator
    def run(self):
        # Input
        inp = self.input()
        seg = inp["seg"]
        seg.open()
        problem = inp["problem"]
        costs = problem.read("costs")
        nodes2blocks = inp["nodes2blocks"].read()

        graph = nifty.graph.UndirectedGraph()
        graph.deserialize(problem.read("graph"))

        workflow_logger.info("TestSubSolver: Starting extraction of subproblems.")
        subproblems = self._run_subproblems_extraction(seg, graph, nodes2blocks)

        workflow_logger.info("TestSubSolver: Starting solvers for subproblems.")
        self._solve_subproblems(costs, subproblems)

        seg.close()

    def _run_subproblems_extraction(self, seg, graph, nodes2blocks):

        # function for subproblem extraction
        # extraction only for level 0
        def extract_subproblem(blockId):
            node_list = nodes2blocks[blockId]
            inner_edges, outer_edges, subgraph = graph.extractSubgraphFromNodes(node_list.tolist())
            return np.array(inner_edges), np.array(outer_edges), subgraph

        blocking = nifty.tools.blocking(roiBegin=[0L, 0L, 0L], roiEnd=seg.shape(), blockShape=self.blockShape)
        number_of_blocks = blocking.numberOfBlocks
        # sample block-ids corresponding to the number of threads
        n_threads = PipelineParameter().nThreads
        sampled_blocks = np.random.choice(number_of_blocks, min(n_threads, number_of_blocks), replace=False)

        # parallel
        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as executor:
            tasks = [executor.submit(extract_subproblem, block_id) for block_id in sampled_blocks]
            sub_problems = [task.result() for task in tasks]

        assert len(sub_problems) == len(sampled_blocks), "%i, %i" % (len(sub_problems), len(sampled_blocks))
        return sub_problems

    def _solve_subproblems(self, costs, sub_problems):

        def _test_mc(g, costs, sub_solver_type):

            if sub_solver_type in ('fm-ilp', 'fm-kl'):
                solver_params  = dict(
                    sigma=PipelineParameter().multicutSigmaFusion,
                    number_of_iterations=PipelineParameter().multicutNumIt,
                    n_stop=PipelineParameter().multicutNumItStopGlobal,
                    n_threads=0,
                    n_fuse=PipelineParameter().multicutNumFuse,
                    seed_fraction=PipelineParameter().multicutSeedFraction
                )
            else:
                solver_params = dict()

            obj = nifty.graph.optimization.multicut.multicutObjective(g, costs)
            solver = string_to_factory(obj, sub_solver_type, solver_params).create(obj)
            t_inf  = time.time()
            res    = solver.optimize()
            t_inf  = time.time() - t_inf
            return obj.evalNodeLabels(res), t_inf

        workflow_logger.info("TestSubSolver: Running sub-block tests for %i blocks" % len(sub_problems))
        available = available_factorys()
        results = {}

        for sub_solver_type in available:
            with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as executor:
                tasks = [executor.submit(
                    _test_mc,
                    sub_problem[2],
                    costs[sub_problem[0]],
                    sub_solver_type) for sub_problem in sub_problems]
                sub_results = [task.result() for task in tasks]
                mean_energy = np.mean([rr[0] for rr in sub_results])
                mean_time   = np.mean([rr[1] for rr in sub_results])
                results[sub_solver_type] = (mean_energy, mean_time)

        for_serialization = []
        for sub_solver_type in available:
            res = results[sub_solver_type]
            workflow_logger.info(
                "TestSubSolver: Result of %s: mean-energy: %f, mean-inference-time: %f"
                % (sub_solver_type, res[0], res[1])
            )
            for_serialization.append([res[0], res[1]])

        self.output().write(available, 'solvers')
        self.output().write(np.array(for_serialization), 'results')

    def output(self):
        blcksize_str = "_".join(map(str, list(self.blockShape)))
        save_name = "TestSubSolver_%s_%s.h5" \
            % (blcksize_str, "modified" if PipelineParameter().defectPipeline else "standard")
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)
