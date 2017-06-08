# Multicut Pipeline implemented with luigi
# Blockwise solver tasks

import luigi

from pipelineParameter import PipelineParameter
from dataTasks import ExternalSegmentation
from customTargets import HDF5DataTarget
from defectDetectionTasks import DefectSliceDetection
from multicutProblemTasks import MulticutProblem

from tools import config_logger, run_decorator, get_unique_rows, find_matching_row_indices
from nifty_helper import run_nifty_solver, string_to_factory, available_factorys

import os
import logging
import time

import numpy as np
import vigra
from concurrent import futures

# import the proper nifty version
try:
    import nifty
    ilp_backend = 'cplex'
except ImportError:
    try:
        import nifty_with_cplex as nifty
        ilp_backend = 'cplex'
    except ImportError:
        import nifty_with_gurobi as nifty
        ilp_backend = 'gurobi'

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


# parent class for blockwise solvers
# TODO move basic functionality from BlockwiseMulticutSolver here
class BlockwiseSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    globalProblem  = luigi.TaskParameter()
    numberOflevels = luigi.Parameter()

    def requires(self):
        # block size in first hierarchy level
        initialBlockShape = PipelineParameter().multicutBlockShape
        # block overlap, for now same for each hierarchy lvl
        blockOverlap = PipelineParameter().multicutBlockOverlap

        problemHierarchy = [self.globalProblem]
        blockFactor = 1

        for l in xrange(self.numberOflevels):
            levelBlockShape = map(lambda x: x * blockFactor, initialBlockShape)

            # TODO check that we don't get larger than the actual shape here
            problemHierarchy.append(
                ReducedProblem(self.pathToSeg, problemHierarchy[-1], levelBlockShape, blockOverlap, l)
            )
            blockFactor *= 2

        return problemHierarchy

    def run(self):
        raise NotImplementedError("BlockwiseSolver is abstract and does not implement a run functionality!")

    # map back to the global solution
    def map_node_result_to_global(self, reduced_node_result):

        problems = self.input()
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

        solver_type = 'fm-kl'  # we use fm with kl as backend, because this shows the best scaling behaviour
        inf_params  = dict(
            sigma=PipelineParameter().multicutSigmaFusion,
            number_of_iterations=PipelineParameter().multicutNumIt,
            n_stop=PipelineParameter().multicutNumItStopGlobal,
            n_threads=PipelineParameter().multicutNThreadsGlobal,
            n_fuse=PipelineParameter().multicutNumFuse,
            seed_fraction=PipelineParameter().multicutSeedFractionGlobal
        )
        factory = string_to_factory(reduced_objective, solver_type, inf_params)
        reduced_node_result, energy, t_inf = run_nifty_solver(
            reduced_objective,
            factory,
            verbose=True,
            time_limit=PipelineParameter().multicutGlobalTimeLimit
        )

        workflow_logger.info(
            "BlockwiseMulticutSolver: Inference of reduced problem for the whole volume took: %f s"
            % (time.time() - t_inf,)
        )

        # NOTE: we don't need to project back to global problem to calculate the correct energy !
        workflow_logger.info(
            "BlockwiseMulticutSolver: Problem solved with energy %f"
            % reduced_objective.evalNodeLabels(reduced_node_result)
        )

        node_result = self.map_node_result_to_global(reduced_node_result)
        self.output().write(node_result)

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "BlockwiseMulticutSolver_%s.h5"
            % ("modified" if PipelineParameter().defectPipeline else "standard",)
        )
        return HDF5DataTarget(save_path)


# stitch blockwise sub-results
class BlockwiseStitchingSolver(BlockwiseSolver):

    @run_decorator
    def run(self):
        problems = self.input()
        reduced_problem = problems[-1]

        reduced_graph = nifty.graph.UndirectedGraph()
        reduced_graph.deserialize(reduced_problem.read("graph"))
        reduced_costs = reduced_problem.read("costs")
        outer_edges = reduced_problem.read("outer_edges")
        uv_ids = reduced_graph.uvIds()

        # merge all edges along the block boundaries that are attractive TODO (modulu bias?!)
        merge_ids = outer_edges[reduced_costs[outer_edges] < 0]

        ufd = nifty.ufd.ufd(reduced_graph.numberOfNodes)
        ufd.merge(uv_ids[merge_ids])

        reduced_node_result = ufd.elementLabeling()

        node_result = self.map_node_result_to_global(reduced_node_result)
        self.output().write(node_result)

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "BlockwiseStitchingSolver_%s.h5"
            % ("modified" if PipelineParameter().defectPipeline else "standard",)
        )
        return HDF5DataTarget(save_path)


# TODO benchmark and speed up
class ReducedProblem(luigi.Task):

    pathToSeg = luigi.Parameter()
    problem   = luigi.TaskParameter()

    blockShape     = luigi.ListParameter()
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
        workflow_logger.info("ReducedProblem: Serializing ndoe converters took: %f s" % (time.time() - t_serialize))

        # find new outer edges
        t_outer = time.time()
        self.find_new_outer_edges(uv_ids, uv_ids_new, old2new_nodes)
        workflow_logger.info("ReducedProblem: Finding new outer edges took: %f s" % (time.time() - t_outer))

        workflow_logger.info("ReucedProblem: Nodes: From %i to %i" % (g.numberOfNodes, number_of_new_nodes))
        workflow_logger.info("ReucedProblem: Edges: From %i to %i" % (g.numberOfEdges, len(uv_ids_new)))

    def find_new_edges_and_costs(self, uv_ids, problem, cut_edges, number_of_new_nodes, old2new_nodes):

        costs  = problem.read("costs")

        ##############################################
        # find new edges and edge weights (vectorized)
        ##############################################
        uv_ids_new = np.sort(old2new_nodes[uv_ids[cut_edges == 1]], axis=1)
        uv_ids_new, inverse_idx = get_unique_rows(uv_ids_new, return_inverse=True)

        number_of_new_edges = uv_ids_new.shape[0]

        costs = costs[cut_edges == 1]
        new_costs = np.zeros(number_of_new_edges, dtype='float32')
        assert inverse_idx.shape[0] == costs.shape[0]

        # TODO vectorize this too
        for i, inv_idx in enumerate(inverse_idx):
            new_costs[inv_idx] += costs[i]

        assert new_costs.shape[0] == number_of_new_edges
        reduced_graph = nifty.graph.UndirectedGraph(number_of_new_nodes)
        reduced_graph.insertEdges(uv_ids_new)

        out = self.output()
        out.write(reduced_graph.serialize(), "graph")
        out.write(reduced_graph.numberOfNodes, 'number_of_nodes')
        out.write(new_costs, "costs")

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

    # find new outer edges by mapping the uv-ids of the outer edges
    # to the new node ids and then find the matching indices of the
    # new uv-ids
    def find_new_outer_edges(self, uv_ids, uv_ids_new, old2new_nodes):
        outer_edges = self.input["sub_solution"].read("outer_edges")
        outer_uvs = np.sort(
            old2new_nodes[uv_ids[outer_edges]],
            axis=1
        )
        new_outer_edges = find_matching_row_indices(uv_ids_new, outer_uvs)[:, 0]
        self.output.write(new_outer_edges, 'outer_edges')

    def output(self):
        blcksize_str = "_".join(map(str, list(self.blockShape)))
        save_name = "ReducedProblem_%s_%s.h5" \
            % (blcksize_str, "modified" if PipelineParameter().defectPipeline else "standard")
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


class NodesToInitialBlocks(luigi.Task):

    pathToSeg = luigi.Parameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()
    dtype        = luigi.ListParameter('uint32')

    def requires(self):
        if PipelineParameter().defectPipeline:
            return {"seg": ExternalSegmentation(self.pathToSeg),
                    "defect_slices": DefectSliceDetection(self.pathToSeg)}
        else:
            return {"seg": ExternalSegmentation(self.pathToSeg)}

    @run_decorator
    def run(self):

        inp = self.input()
        seg = inp["seg"]
        seg.open()

        if PipelineParameter().defectPipeline:
            defect_slices = vigra.readHDF5(inp["defect_slices"].path, 'defect_slices').astype('int64').tolist()
        else:
            defect_slices = []

        blocking = nifty.tools.blocking(roiBegin=[0L, 0L, 0L], roiEnd=seg.shape(), blockShape=self.blockShape)
        number_of_blocks = blocking.numberOfBlocks
        block_overlap = list(self.blockOverlap)

        n_workers = min(number_of_blocks, PipelineParameter().nThreads)
        # nWorkers = 1
        block_result = nifty.tools.nodesToBlocksStacked(seg.get(), blocking, block_overlap, defect_slices, n_workers)

        block_result = [np.array(b_res, dtype=self.dtype) for b_res in block_result]
        self.output().writeVlen(block_result)

    def output(self):
        save_name = "NodesToInitialBlocks.h5"
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


class BlockwiseSubSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    problem   = luigi.TaskParameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()

    level = luigi.Parameter()

    def requires(self):
        initialShape = PipelineParameter().multicutBlockShape
        overlap      = PipelineParameter().multicutBlockOverlap

        nodes2blocks = NodesToInitialBlocks(self.pathToSeg, initialShape, overlap)
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
            return np.array(inner_edges), np.array(outer_edges), subgraph

        block_overlap = list(self.blockOverlap)
        blocking = nifty.tools.blocking(roiBegin=[0L, 0L, 0L], roiEnd=seg.shape(), blockShape=self.blockShape)
        number_of_blocks = blocking.numberOfBlocks

        # sequential for debugging
        # subProblems = []
        # for blockId in xrange(numberOfBlocks):
        #     print "Running block:", blockId, "/", numberOfBlocks
        #     t_block = time.time()

        #     block = blocking.getBlockWithHalo(blockId, blockOverlap).outerBlock
        #     blockBegin, blockEnd = block.begin, block.end
        #     workflow_logger.debug(
        #       "Block id " + str(blockId) + " start " + str(blockBegin) + " end " + str(blockEnd)
        #     )
        #     subBlocks = initialBlocking.getBlockIdsInBoundingBox(blockBegin, blockEnd, initialOverlap)
        #     subProblems.append( extract_subproblem( blockId, subBlocks ) )

        n_workers = min(number_of_blocks, PipelineParameter().nThreads)
        # n_workers = 2
        # parallel
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = []
            for block_id in xrange(number_of_blocks):
                block = blocking.getBlockWithHalo(block_id, block_overlap).outerBlock
                block_begin, block_end = block.begin, block.end
                sub_blocks = initial_blocking.getBlockIdsInBoundingBox(block_begin, block_end, initial_overlap)
                workflow_logger.debug(
                    "BlockwiseSubSolver: block id %i start %s end %s" % (block_id, str(block_begin), str(block_end))
                )
                tasks.append(executor.submit(extract_subproblem, block_id, sub_blocks))
            sub_problems = [task.result() for task in tasks]

        # we need to serialize the outer edges for the stitching solvers
        self.output().write(
            np.unique(np.concatenate([sub_prob[1] for sub_prob in sub_problems])),
            'outer_edges'
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

        def _solve_mc(g, costs, block_id):
            workflow_logger.debug(
                "BlockwiseSubSolver: Solving MC Problem with %i / %i number of variables"
                % (g.numberOfNodes, g.numberOfEdges)
            )
            obj = nifty.graph.optimization.multicut.multicutObjective(g, costs)
            factory = string_to_factory(obj, sub_solver_type, solver_params)
            solver = factory.create(obj)
            t_inf  = time.time()
            res    = solver.optimize()
            workflow_logger.debug(
                "BlockwiseSubSolver: Inference for block %i with fusion moves solver in %f s"
                % (block_id, t_inf - time.time())
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
        cut_edges[cut_edges >= 1] = 1
        self.output().write(cut_edges)

    def output(self):
        blcksize_str = "_".join(map(str, list(self.blockShape)))
        save_name = "BlockwiseSubSolver_%s_%s.h5" \
            % (blcksize_str, "modified" if PipelineParameter().defectPipeline else "standard")
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


# only works for level 1 for now!
class TestSubSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToClassifier = luigi.Parameter()

    blockShape   = luigi.ListParameter(default=PipelineParameter().multicutBlockShape)
    blockOverlap = luigi.ListParameter(default=PipelineParameter().multicutBlockOverlap)

    def requires(self):
        nodes2blocks = NodesToInitialBlocks(self.pathToSeg, self.blockShape, self.blockOverlap)
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
