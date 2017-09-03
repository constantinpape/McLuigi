

# Multicut Pipeline implemented with luigi
# Blockwise solver tasks

import luigi

from .pipelineParameter import PipelineParameter
from .dataTasks import ExternalSegmentation, StackedRegionAdjacencyGraph
from .customTargets import HDF5DataTarget, FolderTarget
from .defectDetectionTasks import DefectSliceDetection
from .blocking_helper import EdgesBetweenBlocks, BlockGridGraph
from .blockwiseMulticutTasks import BlockwiseSolver, BlockwiseSubSolver

from .nifty_helper import run_nifty_solver, string_to_factory
from .tools import config_logger, run_decorator, get_replace_slices, replace_from_dict

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
        # for block_id in range(1):
        for block_id in range(len(sub_results)):
            sub_result = {sub_nodes[block_id][i]: sub_results[block_id][i]
                          for i in range(len(sub_nodes[block_id]))}

            print("Saving Block-Result for block %i / %i" % (block_id, len(sub_results)))
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
                block_begins[block_id],
                block_ends[block_id]
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
                            [z, 0, 0],
                            out_array.readSubarray([replace_z, 0, 0], [replace_z + 1, block_shape[1], block_shape[2]])
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


# stitch blockwise sub-results according to costs of the edges connecting the sub-blocks
class BlockwiseStitchingSolver(BlockwiseSolver):

    boundaryBias = luigi.Parameter(default=.5)

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
        outer_edges = np.unique(
            np.concatenate([out_edges for out_edges in outer_edges])
        )

        workflow_logger.info(
            "BlockwiseStitchingSolver: Looking for merge edges in %i between block edges of %i total edges"
            % (len(outer_edges), len(uv_ids))
        )

        # merge all edges along the block boundaries that are attractive (energy bigger than bias)
        energyBias = 0. if self.boundaryBias == .5 else \
            np.log((1. - self.boundaryBias) / self.boundaryBias)
        merge_ids = outer_edges[reduced_costs[outer_edges] > energyBias]

        workflow_logger.info(
            "BlockwiseStitchingSolver: Merging %i edges with value larger than bias %f of %i between block edges"
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
        save_name = "BlockwiseStitchingSolver_L%i_%s_%s_%s_%.2f.h5" % (
            self.numberOfLevels,
            '_'.join(map(str, PipelineParameter().multicutBlockShape)),
            '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
            "modified" if PipelineParameter().defectPipeline else "standard",
            self.boundaryBias
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


# stitch blockwise sub-results by running Multicut only on the edges between blocks
# -> no idea if this will actually work
# -> two options:
# --> run Multicut on the block overlaps
# --> run Multicuts for all pairs of adjacent blocks and check edges that are part of multiple
#     block pairs for consistency -> if inconsistent, don't merge
# -> maybe this needs a different problem formulation than Multicut ?!

class BlockwiseMulticutStitchingSolver(BlockwiseSolver):

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
            'seg': ExternalSegmentation(self.pathToSeg),
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
        inp = self.input()
        problems = inp['problems']
        reduced_problem = problems[-1]
        global2new_nodes = reduced_problem.read('global2new')

        edges_between_blocks = inp['edges_between_blocks'].read('edges_between_blocks')
        block_uv_ids = inp['edges_between_blocks'].read('block_uv_ids')

        seg = inp['seg']
        seg.open()

        # load the reduced graph of the current level
        reduced_graph = nifty.graph.UndirectedGraph()
        reduced_graph.deserialize(reduced_problem.read("graph"))
        reduced_costs = reduced_problem.read("costs")
        assert len(reduced_costs) == reduced_graph.numberOfEdges
        reduced_objective = nifty.graph.optimization.multicut.multicutObjective(reduced_graph, reduced_costs)

        t_extract = time.time()
        sub_problems = self._extract_subproblems(reduced_graph, block_uv_ids, seg, global2new_nodes, edges_between_blocks)
        workflow_logger.info(
            "BlockwiseMulticutStitchingSolver: Problem extraction took %f s" % (time.time() - t_extract)
        )

        t_solve = time.time()
        edge_results = self._solve_subproblems(sub_problems, reduced_costs)
        workflow_logger.info("BlockwiseMulticutStitchingSolver: Problem solving took %f s" % (time.time() - t_solve))

        t_merge = time.time()
        reduced_node_result = self._merge_blocks(reduced_graph, edge_results)
        workflow_logger.info("BlockwiseMulticutStitchingSolver: Merging took %f s" % (time.time() - t_merge))

        workflow_logger.debug(
            "BlockwiseMulticutStitchingSolver: Number of nodes after merging: %i / %i"
            % (len(np.unique(reduced_node_result)), reduced_graph.numberOfNodes)
        )

        workflow_logger.info(
            "BlockwiseMulticutStitchingSolver: Problem solved with energy %f"
            % reduced_objective.evalNodeLabels(reduced_node_result)
        )

        node_result = self.map_node_result_to_global(problems, reduced_node_result)
        assert len(np.unique(node_result)) == len(np.unique(reduced_node_result))
        self.output().write(node_result)
        seg.close()

    def _extract_subproblems(self, reduced_graph, block_uv_ids, seg, global2new_nodes, edges_between_blocks):

        uv_ids = reduced_graph.uvIds()

        # construct the blocking for the current block size
        # get the overlap
        overlap = list(PipelineParameter().multicutBlockOverlap)

        # get the block shape of the current level
        initial_shape = PipelineParameter().multicutBlockShape
        block_factor  = max(1, (self.numberOfLevels - 1) * 2)
        block_shape = list(map(lambda x: x * block_factor, initial_shape))
        blocking = nifty.tools.blocking(
            roiBegin=[0, 0, 0],
            roiEnd=seg.shape(),
            blockShape=block_shape
        )

        def extract_subproblem(block_edge_id, block_uv):

            block_u, block_v = block_uv

            # find the actual overlapping regions in block u and v and load them
            have_overlap, ovlp_begin_u, ovlp_end_u, ovlp_begin_v, ovlp_end_v = blocking.getLocalOverlaps(block_u, block_v, overlap)

            outer_block_u = blocking.getBlockWithHalo(block_u, overlap).outerBlock
            begin_u, end_u = outer_block_u.begin, outer_block_u.end

            # make sure that they are indeed overlapping
            if not have_overlap:
                v_block = blocking.getBlockWithHalo(block_v, overlap).outerBlock
                v_begin, v_end = v_block.begin, v_block.end
                raise RuntimeError(
                    "No overlap found for blocks %i, %i with coords %s, %s and %s, %s" % (block_u, block_v, str(u_begin), str(u_end), str(v_begin), str(v_end))
                )

            # read the segmentation in the ovlp get the nodes and transform them to current node ids
            ovlp_begin = (np.array(ovlp_begin_u) + np.array(begin_u)).tolist()
            ovlp_end   = (np.array(ovlp_end_u) + np.array(begin_u)).tolist()
            seg_ovlp = seg.read(ovlp_begin, ovlp_end)

            # debug views
            if False and block_edge_id != 0:
                from volumina_viewer import volumina_n_layer
                rawp = PipelineParameter().inputs['data'][0]
                bb = np.s_[
                    ovlp_begin[0]:ovlp_end[0],
                    ovlp_begin[1]:ovlp_end[1],
                    ovlp_begin[2]:ovlp_end[2],
                ]
                with h5py.File(rawp) as f:
                    raw = f['data'][bb].astype('float32')
                volumina_n_layer([raw, seg_ovlp])
                quit()

            workflow_logger.debug(
                "BlockwiseMulticutStitchingSolver: Extracting problem from block-edge %i ovlp between block %i, %i with shape %s"
                % (block_edge_id, block_u, block_v, str(seg_ovlp.shape))
            )

            reduced_nodes = np.unique(global2new_nodes[np.unique(seg_ovlp)])

            # add nodes from the outer edges that go between these blocks
            additional_edges = uv_ids[edges_between_blocks[block_edge_id]]
            assert additional_edges.ndim == 2

            additional_nodes = np.unique(additional_edges)
            reduced_nodes = np.unique(
                np.concatenate([reduced_nodes, additional_nodes])
            )

            # extract the subproblem
            inner_edges, _, subgraph = reduced_graph.extractSubgraphFromNodes(reduced_nodes)
            assert len(inner_edges) == subgraph.numberOfEdges

            workflow_logger.debug(
                "BlockwiseMulticutStitchingSolver: Extracted problem for block-edge %i has %i nodes and %i edges"
                % (block_edge_id, subgraph.numberOfNodes, subgraph.numberOfEdges)
            )
            return subgraph, inner_edges

        return [extract_subproblem(block_edge_id, block_uv) for block_edge_id, block_uv in enumerate(block_uv_ids)]

    def _solve_subproblems(self, sub_problems, reduced_costs):

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

        workflow_logger.info("BlockwiseMulticutStitichingSolver: Solving sub-problems with solver %s" % sub_solver_type)
        workflow_logger.info(
            "BlockwiseMulticutStitichingSolver: With Params %s" % ' '.join(
                ['%s, %s,' % (str(k), str(v)) for k, v in solver_params.items()]
            )
        )

        def mc(graph, costs):
            obj = nifty.graph.optimization.multicut.multicutObjective(graph, costs)
            factory = string_to_factory(obj, sub_solver_type, solver_params)
            solver = factory.create(obj)
            return solver.optimize()

        # solve subproblems in parallel
        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as tp:
            tasks = [tp.submit(mc, prob[0], reduced_costs[prob[1]]) for prob in sub_problems]
            sub_results = [t.result() for t in tasks]

        edge_result = np.zeros(len(reduced_costs), dtype='uint8')
        edge_is_covered = np.zeros(len(reduced_costs), dtype=bool)

        # combine the subproblem results into global edge vector
        for problem_id, sub_prob in enumerate(sub_problems):
            node_result = sub_results[problem_id]
            sub_uv_ids = sub_prob[0].uvIds()

            edge_sub_result = node_result[sub_uv_ids[:, 0]] != node_result[sub_uv_ids[:, 1]]
            workflow_logger.debug(
                "BlockwiseMulticutStitichingSolver: Number of cut edges for block-edge %i is %i / %i"
                % (problem_id, np.sum(edge_sub_result), len(edge_sub_result))
            )

            edge_result[sub_prob[1]] += edge_sub_result
            edge_is_covered[sub_prob[1]] = True

        # the actual edge result, where all edges that were not covered in one of the sub-problems
        # stay cut
        real_edge_result = np.ones_like(edge_result, dtype='uint8')
        real_edge_result[edge_is_covered] = edge_result[edge_is_covered]

        return real_edge_result

    def _merge_blocks(self, reduced_graph, edge_result):

        ufd = nifty.ufd.ufd(reduced_graph.numberOfNodes)
        uv_ids = reduced_graph.uvIds()

        merge_edges = uv_ids[edge_result == 0]
        workflow_logger.info(
            "BlockwiseMulticutStitichingSolver: Number of edges that will be merged: %i / %i"
            % (len(merge_edges), len(uv_ids))
        )
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


# don't stitch sub-results
# for proper baseline
class NoStitchingSolver(BlockwiseSolver):

    def run(self):
        problems = self.input()
        reduced_problem = problems[-1]
        n_nodes = reduced_problem.read('number_of_nodes')
        reduced_node_result = np.arange(n_nodes, dtype='uint32')
        node_result = self.map_node_result_to_global(problems, reduced_node_result)
        self.output().write(node_result)

    def output(self):
        save_name = "NoStitchingSolver_L%i_%s_%s_%s.h5" % (
            self.numberOfLevels,
            '_'.join(map(str, PipelineParameter().multicutBlockShape)),
            '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
            "modified" if PipelineParameter().defectPipeline else "standard"
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


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
            'sub_solver': sub_solver,
            'seg': ExternalSegmentation(self.pathToSeg)
        }

    @run_decorator
    def run(self):

        # get all inputs
        inp = self.input()
        subblocks = inp['subblocks']
        problems = inp['problems']
        block_graph = nifty.graph.UndirectedGraph()
        # we only consider blocks that are in 6 adjacency
        block_graph.deserialize(inp['block_graph'].read('nearest_neighbors'))
        sub_solver = inp['sub_solver']

        # get the shape
        seg = inp['seg']
        seg.open()
        shape = seg.shape()
        seg.close()

        # read the relevant problem, which is the second to last reduced problem ->
        # because we merge according to results of last BlockwiseSubSolver
        reduced_problem = problems[-2]
        reduced_graph = nifty.graph.UndirectedGraph()
        reduced_graph.deserialize(reduced_problem.read("graph"))
        reduced_costs = reduced_problem.read("costs")
        reduced_objective = nifty.graph.optimization.multicut.multicutObjective(reduced_graph, reduced_costs)

        # find the node overlaps
        t_ovlp = time.time()
        node_overlaps = self._find_node_overlaps(subblocks, block_graph, shape)
        workflow_logger.info(
            "BlockwiseOverlapSolver: extracting overlapping nodes from blocks in %f s" % (time.time() - t_ovlp,)
        )

        # stitch the blocks based on the node overlaps
        t_stitch = time.time()
        reduced_node_result = self._stitch_blocks(block_graph, node_overlaps, sub_solver, reduced_graph)
        workflow_logger.info(
            "BlockwiseOverlapSolver: stitching blocks in %f s" % (time.time() - t_stitch,)
        )

        # get the energy of the solution
        workflow_logger.info(
            "BlockwiseOverlapSolver: Problem solved with energy %f"
            % reduced_objective.evalNodeLabels(reduced_node_result)
        )

        # map back to the global nodes and write result
        # we only need to do this for nLevels > 1, because for nLevels == 1, we already have the original problem
        # and hence original node ids
        if self.numberOfLevels > 1:
            node_result = self.map_node_result_to_global(problems, reduced_node_result, -2)
        else:
            node_result = reduced_node_result
        self.output().write(node_result)

    def _find_node_overlaps(self, subblocks, block_graph, shape):

        block_res_path = subblocks.path
        # read the nodes from the sub solver to convert back to
        # the actual problem node ids

        # construct the blocking for the current block size
        # get the overlap
        overlap = list(PipelineParameter().multicutBlockOverlap)

        # get the block shape of the current level
        initial_shape = PipelineParameter().multicutBlockShape
        block_factor  = max(1, (self.numberOfLevels - 1) * 2)
        block_shape = list(map(lambda x: x * block_factor, initial_shape))
        blocking = nifty.tools.blocking(
            roiBegin=[0, 0, 0],
            roiEnd=shape,
            blockShape=block_shape
        )

        # extract the overlaps for all the edges
        def node_overlaps_for_block_pair(block_edge_id, block_uv):

            block_u, block_v = block_uv
            # get the uv-ids connecting the two blocks and the paths to the block segmentations
            block_u_path = os.path.join(block_res_path, 'block%i_segmentation.h5' % block_u)
            block_v_path = os.path.join(block_res_path, 'block%i_segmentation.h5' % block_v)

            # find the actual overlapping regions in block u and v and load them
            have_overlap, ovlp_begin_u, ovlp_end_u, ovlp_begin_v, ovlp_end_v = blocking.getLocalOverlaps(block_u, block_v, overlap)
            if not have_overlap:
                u_block = blocking.getBlockWithHalo(block_u, overlap).outerBlock
                v_block = blocking.getBlockWithHalo(block_v, overlap).outerBlock
                u_begin, u_end = u_block.begin, u_block.end
                v_begin, v_end = v_block.begin, v_block.end
                raise RuntimeError(
                    "No overlap found for blocks %i, %i with coords %s, %s and %s, %s" % (block_u, block_v, str(u_begin), str(u_end), str(v_begin), str(v_end))
                )

            overlap_bb_u = np.s_[ovlp_begin_u[0]:ovlp_end_u[0], ovlp_begin_u[1]:ovlp_end_u[1], ovlp_begin_u[2]:ovlp_end_u[2]]
            overlap_bb_v = np.s_[ovlp_begin_v[0]:ovlp_end_v[0], ovlp_begin_v[1]:ovlp_end_v[1], ovlp_begin_v[2]:ovlp_end_v[2]]

            with h5py.File(block_u_path) as f_u, \
                    h5py.File(block_v_path) as f_v:

                seg_u = f_u['data'][overlap_bb_u]
                seg_v = f_v['data'][overlap_bb_v]

            # debugging view
            if False:
                from volumina_viewer import volumina_n_layer
                u_block = blocking.getBlockWithHalo(block_u, overlap).outerBlock
                raw_path = PipelineParameter().inputs['data'][0]
                oseg_path = PipelineParameter().inputs['seg']
                u_begin, u_end = u_block.begin, u_block.end
                with h5py.File(raw_path) as f_raw, \
                    h5py.File(oseg_path) as f_oseg:
                    global_bb = np.s_[
                        ovlp_begin_u[0] + u_begin[0]:ovlp_end_u[0] + u_begin[0],
                        ovlp_begin_u[1] + u_begin[1]:ovlp_end_u[1] + u_begin[1],
                        ovlp_begin_u[2] + u_begin[2]:ovlp_end_u[2] + u_begin[2]
                    ]
                    raw = f_raw['data'][global_bb].astype('float32')
                    oseg = f_oseg['data'][global_bb]
                volumina_n_layer([raw, oseg, seg_u, seg_v])
                quit()

            nodes_u = np.unique(seg_u)
            nodes_v = np.unique(seg_v)
            # find the overlaps between the two segmentations
            # NOTE: nodes_u is not dense, I don't know if this is much of a performance issue, but I
            # really don't want to do all the mapping to make it dense

            # get the overlap counters
            overlap_counter_u = ngt.Overlap(nodes_u[-1], seg_u, seg_v)
            overlap_counter_v = ngt.Overlap(nodes_v[-1], seg_v, seg_u)

            # FIXME this looks very inefficient....
            # if we ever want to put this into production, move to c++
            overlaps_u = {}
            # calculate the symmetrical overlaps
            for node_u in nodes_u:
                ovlp_nodes, rel_ovlps = overlap_counter_u.overlapArraysNormalized(node_u)

                symmetrical_overlaps = np.zeros_like(rel_ovlps, dtype=rel_ovlps.dtype)
                for ii, node_v in enumerate(ovlp_nodes):
                    ovlp_u = rel_ovlps[ii]
                    ovlp_nodes_v, rel_ovlps_v = overlap_counter_v.overlapArraysNormalized(node_v)
                    where_node_u = ovlp_nodes_v == node_u
                    ovlp_v = rel_ovlps_v[where_node_u]

                    # TODO for now we just average the relative overlaps, but maybe it would be a better idea to use min
                    symmetrical_overlaps[ii] = (ovlp_u + ovlp_v) / 2.

                overlaps_u[node_u] = (ovlp_nodes, symmetrical_overlaps)

            return overlaps_u

        # serial for debugging
        node_overlaps = [node_overlaps_for_block_pair(block_edge_id, block_uv) for block_edge_id, block_uv in enumerate(block_graph.uvIds())]

        # parallel
        # with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as tp:
        #     tasks = [
        #         tp.submit(node_overlaps_for_block_pair, block_edge_id, block_uv)
        #         for block_edge_id, block_uv in enumerate(block_graph.uvIds())
        #     ]
        #     node_overlaps = [t.result() for t in tasks]

        return node_overlaps

    def _stitch_blocks(self, block_graph, node_overlaps, sub_solver, reduced_graph):

        n_blocks = block_graph.numberOfNodes
        # read the sub_node results
        sub_node_results = sub_solver.read('sub_results')
        assert len(sub_node_results) == n_blocks

        # apply offset to each sub-block result to have unique ids before stitching
        offset = 0
        block_offsets = []
        for sub_res in sub_node_results:
            block_offsets.append(offset)
            offset += np.max(sub_res) + 1

        block_offsets = np.array(block_offsets, dtype='uint32')
        offset = int(offset)

        # create ufd to merge with last offset value -> number of nodes that need to be merged
        ufd = nifty.ufd.ufd(offset)

        # now, we iterate over the block pairs and merge nodes according to their overlap
        for block_pair_id, block_uv in enumerate(block_graph.uvIds()):
            block_u, block_v = block_uv

            # get the results from the overlap calculations
            this_node_overlaps = node_overlaps[block_pair_id]
            assert len(this_node_overlaps) <= len(sub_node_results[block_u]), "%i, %i" % (len(this_node_overlaps), len(sub_node_results[block_u]))

            offsets_u = block_offsets[block_u]
            offsets_v = block_offsets[block_v]

            # iterate over the nodes in overlap(u) and merge with nodes in overlap(v)
            # according to the overlaps
            # we only merge with the max overlap node, if the relative overlap is bigger then the threshold
            for node_u_id, node_ovlp in this_node_overlaps.items():
                # nodes are sorted in ascending order of their relative overlap
                relative_ovlp = node_ovlp[1][-1]
                if relative_ovlp > PipelineParameter().overlapThreshold:
                    merge_node_v = node_ovlp[0][-1]
                    ufd.merge(node_u_id + offsets_u, merge_node_v + offsets_v)

        # get the merge result
        node_result = ufd.elementLabeling()
        assert len(node_result) == offset, "%i, %i" % (len(node_result), offset)

        # project back to the reduced problem nodes via iterating over the blocks and projection of
        # the block node result
        reduced_node_result = np.zeros(reduced_graph.numberOfNodes, dtype='uint32')
        sub_nodes = sub_solver.read('sub_nodes')

        for block_id in range(n_blocks):

            # first find the result for the nodes in this block
            block_result = node_result[block_offsets[block_id]:block_offsets[block_id + 1]] \
                if block_id < n_blocks - 1 else node_result[block_offsets[block_id]:offset]

            # next, map the merge result to the reduced-problem nodes
            sub_result = sub_node_results[block_id]
            reduced_result = block_result[sub_result]

            # finally, write to the block result
            reduced_node_result[sub_nodes[block_id]] = reduced_result

        return reduced_node_result

    def output(self):
        save_name = "BlockwiseOverlapSolver_L%i_%s_%s_%s_%.2f.h5" % (
            self.numberOfLevels,
            '_'.join(map(str, PipelineParameter().multicutBlockShape)),
            '_'.join(map(str, PipelineParameter().multicutBlockOverlap)),
            "modified" if PipelineParameter().defectPipeline else "standard",
            PipelineParameter().overlapThreshold
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)
