from __future__ import division, print_function

# Multicut Pipeline implemented with luigi
# Blockwise solver tasks

import luigi

from pipelineParameter import PipelineParameter
from dataTasks import ExternalSegmentation, StackedRegionAdjacencyGraph
from customTargets import HDF5DataTarget, FolderTarget
from defectDetectionTasks import DefectSliceDetection
from multicutProblemTasks import MulticutProblem
from blocking_helper import NodesToBlocks

from tools import config_logger, run_decorator
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
except ImportError:
    try:
        import nifty_with_cplex as nifty
    except ImportError:
        import nifty_with_gurobi as nifty

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
    def map_node_result_to_global(self, problems, reduced_node_result, reduced_problem_index=-1):

        n_nodes_global = problems[0].read('number_of_nodes')
        reduced_problem = problems[reduced_problem_index]
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

        # we set visit-nth to 1 for the fusion move solvers and to 100 for kernighan lin
        # NOTE: we will not use ilp here, so it does not matter that it is handled incorrectly
        visit_nth = 1 if solver_type.startswith('fm') else 100

        factory = string_to_factory(reduced_objective, solver_type, inf_params)
        reduced_node_result, energy, t_inf = run_nifty_solver(
            reduced_objective,
            factory,
            verbose=1,
            time_limit=PipelineParameter().multicutGlobalTimeLimit,
            visit_nth=visit_nth
        )
        workflow_logger.info(
            "BlockwiseMulticutSolver: Inference of reduced problem for the whole volume took: %f s" % t_inf[-1]
        )

        workflow_logger.info(
            "BlockwiseMulticutSolver: Problem solved with energy %f"
            % energy[-1]
        )

        # TODO change to debug
        if workflow_logger.isEnabledFor(logging.INFO):
            assert len(energy) == len(t_inf)
            workflow_logger.info(
                "BlockwiseMulticutSolver: logging energy during inference:"
            )
            for ii in xrange(len(energy)):
                workflow_logger.info(
                    "BlockwiseMulticutSolver: t: %f s energy: %f" % (t_inf[ii], energy[ii])
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
        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as tp:
            tasks = [tp.submit(edge_mapping.getNewEdgeIds, oeids) for oeids in outer_edge_ids]
            new_outer_edges = np.array([np.array(t.result()) for t in tasks])

        assert len(new_costs) == len(uv_ids_new)
        reduced_graph = nifty.graph.UndirectedGraph(number_of_new_nodes)
        reduced_graph.insertEdges(uv_ids_new)

        out = self.output()
        out.write(reduced_graph.serialize(), "graph")
        out.write(reduced_graph.numberOfNodes, 'number_of_nodes')
        out.write(new_costs, "costs")
        out.writeVlen(new_outer_edges, 'outer_edges')

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
    serializeSubResults = luigi.Parameter(default=True)
    # will outer edges be cut ?
    # should be left at true, because results seem to degraded if false
    cutOuterEdges = luigi.Parameter(default=True)

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
        out.writeVlen(
            np.array([sub_prob[1] for sub_prob in sub_problems]),
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
            if self.cutOuterEdges:
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
