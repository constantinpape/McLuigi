# Multicut Pipeline implemented with luigi
# Blockwise solver tasks

import luigi

from pipelineParameter import PipelineParameter
from dataTasks import ExternalSegmentation
from customTargets import HDF5DataTarget
from defectDetectionTasks import DefectSliceDetection
from multicutProblemTasks import MulticutProblem

from tools import config_logger, run_decorator
from nifty_helper import run_nifty_solver, string_to_factory, available_factorys

import os
import logging
import json
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


# produce reduced global graph from subproblems
# solve global multicut problem on the reduced graph
class BlockwiseMulticutSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    globalProblem  = luigi.TaskParameter()

    numberOflevels = luigi.Parameter()

    def requires(self):
        # block size in first hierarchy level
        initialBlockShape = PipelineParameter().multicutBlockShape
        # block overlap, for now same for each hierarchy lvl
        blockOverlap = PipelineParameter().multicutBlockOverlap

        problemHierarchy = [self.globalProblem,]
        blockFactor = 1

        for l in xrange(self.numberOflevels):
            levelBlockShape = map( lambda x: x*blockFactor, initialBlockShape)

            workflow_logger.info("BlockwiseMulticutSolver: scheduling reduced problem for level %i with block shape: %s" % (l, str(levelBlockShape)) )

            # TODO check that we don't get larger than the actual shape here
            problemHierarchy.append( ReducedProblem(self.pathToSeg, problemHierarchy[-1],
                levelBlockShape , blockOverlap, l))
            blockFactor *= 2

        return problemHierarchy

    @run_decorator
    def run(self):

        problems = self.input()

        n_nodes_global = problems[0].read('number_of_nodes')
        # we solve the problem for the costs and the edges of the last level of hierarchy
        reduced_problem = problems[-1]

        reduced_graph = nifty.graph.UndirectedGraph()
        reduced_graph.deserialize( reduced_problem.read("graph") )
        reduced_costs = reduced_problem.read("costs")
        reduced_objective = nifty.graph.optimization.multicut.multicutObjective(reduced_graph, reduced_costs)

        # get solver and run global inference
        solver_type = 'fm-kl'
        inf_params  = dict(sigma = PipelineParameter().multicutSigmaFusion,
                number_of_iterations = PipelineParameter().multicutNumIt,
                n_stop = PipelineParameter().multicutNumItStopGlobal,
                n_threads = PipelineParameter().multicutNThreadsGlobal,
                n_fuse    = PipelineParameter().multicutNumFuse,
                seed_fraction = PipelineParameter().multicutSeedFractionGlobal
                )
        factory = string_to_factory(reduced_objective, solver_type, inf_params)
        reduced_node_result, energy, t_inf = run_nifty_solver(reduced_objective, factory,
                verbose = True,
                time_limit = PipelineParameter().multicutGlobalTimeLimit)

        workflow_logger.info("BlockwiseMulticutSolver: Inference of reduced problem for the whole volume took: %f s" % (time.time() - t_inf,))
        # Note that we don't need to project back to global problem to calculate the correct energy !
        workflow_logger.info("BlockwiseMulticutSolver: Problem solved with energy %f" % reduced_objective.evalNodeLabels(reduced_node_result) )

        to_global_nodes = reduced_problem.read("new2global")

        # TODO vectorize
        node_result = np.zeros(n_nodes_global, dtype = 'uint32')
        for node_id, node_res in enumerate(reduced_node_result):
            node_result[to_global_nodes[node_id]] = node_res

        self.output().write(node_result)

    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "BlockwiseMulticutSolver_%s.h5" % ("modifed" if PipelineParameter().defectPipeline else "standard",))
        return HDF5DataTarget( save_path )


# TODO benchmark and speed up
class ReducedProblem(luigi.Task):

    pathToSeg = luigi.Parameter()
    problem   = luigi.TaskParameter()

    blockShape     = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()

    level = luigi.Parameter()

    def requires(self):
        return {"subSolution" : BlockwiseSubSolver( self.pathToSeg, self.problem, self.blockShape, self.blockOverlap, self.level),
                "problem" : self.problem }

    @run_decorator
    def run(self):

        inp = self.input()
        problem   = inp["problem"]
        cutEdges = inp["subSolution"].read()

        g = nifty.graph.UndirectedGraph()
        g.deserialize(problem.read("graph"))

        numberOfNodes = g.numberOfNodes
        numberOfEdges = g.numberOfEdges

        uvIds = g.uvIds()
        costs  = problem.read("costs")

        t_merge = time.time()
        ufd = nifty.ufd.ufd( numberOfNodes )

        # TODO maybe we could spped this up if we pass a list instead of an np.array!
        mergeNodes = uvIds[cutEdges == 0]
        ufd.merge(mergeNodes)

        old2newNodes = ufd.elementLabeling()
        new2oldNodes = ufd.representativesToSets()
        # number of nodes for the new problem
        numberOfNewNodes = len(new2oldNodes)

        ##############################################
        # find new edges and edge weights (vectorized)
        ##############################################
        uvIdsNew = np.sort( old2newNodes[ uvIds[cutEdges==1] ], axis = 1)

        uvIdsUnique = np.ascontiguousarray(uvIdsNew).view( np.dtype((np.void, uvIdsNew.itemsize * uvIdsNew.shape[1])) )
        _, uniqueIdx, inverseIdx = np.unique(uvIdsUnique, return_index = True, return_inverse = True)
        uvIdsNew = uvIdsNew[uniqueIdx]
        numberOfNewEdges = uvIdsNew.shape[0]

        costs = costs[cutEdges==1]
        newCosts = np.zeros(numberOfNewEdges, dtype = 'float32')
        assert inverseIdx.shape[0] == costs.shape[0]

        # TODO vectorize this too
        for i, invIdx in enumerate(inverseIdx):
            newCosts[invIdx] += costs[i]

        assert newCosts.shape[0] == numberOfNewEdges
        reducedGraph = nifty.graph.UndirectedGraph(numberOfNewNodes)
        reducedGraph.insertEdges(uvIdsNew)

        if self.level == 0:
            global2new = old2newNodes
            new2global = new2oldNodes

        else:
            global2newLast = problem.read("global2new").astype(np.uint32)
            new2globalLast = problem.read("new2global")

            global2new = -1 * np.ones_like( global2newLast, dtype = np.int32)
            new2global = []

            #TODO vectorize
            for newNode in xrange(numberOfNewNodes):
                oldNodes = new2oldNodes[newNode]
                globalNodes = np.concatenate(new2globalLast[oldNodes])
                global2new[ globalNodes ] = newNode
                new2global.append( globalNodes )

            assert not -1 in global2new
            global2new = global2new.astype('uint32')


        t_merge = time.time() - t_merge
        workflow_logger.info("ReucedProblem: Time for merging: %f s" % (t_merge))

        workflow_logger.info("ReucedProblem: Merging of blockwise results reduced problemsize (level %i):" % (self.level,) )
        workflow_logger.info("ReucedProblem: Nodes: From %i to %i" % (numberOfNodes, numberOfNewNodes) )
        workflow_logger.info("ReucedProblem: Edges: From %i to %i" % (numberOfEdges, numberOfNewEdges) )

        out = self.output()
        out.write(reducedGraph.serialize(), "graph")
        out.write(reducedGraph.numberOfNodes, 'number_of_nodes')
        out.write(newCosts, "costs")

        out.write(global2new, "global2new")

        # need to serialize this differently, due to list of list
        new2oldNodes = np.array([np.array(n2o, dtype = 'uint32') for n2o in new2oldNodes])
        out.writeVlen(new2oldNodes, "new2old")

        new2global = np.array([np.array(n2g, dtype = 'uint32') for n2g in new2global])
        out.writeVlen(new2global, "new2global")


    def output(self):
        blcksize_str = "_".join( map( str, list(self.blockShape) ) )
        save_name = "ReducedProblem_%s_%s.h5" % (blcksize_str,"modifed" if PipelineParameter().defectPipeline else "standard")
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )


class NodesToInitialBlocks(luigi.Task):

    pathToSeg = luigi.Parameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()
    dtype        = luigi.ListParameter('uint32')

    def requires(self):
        if PipelineParameter().defectPipeline:
            return {"seg" : ExternalSegmentation(self.pathToSeg),
                    "defect_slices" : DefectSliceDetection(self.pathToSeg)}
        else:
            return {"seg" : ExternalSegmentation(self.pathToSeg) }

    @run_decorator
    def run(self):

        inp = self.input()
        seg = inp["seg"]
        seg.open()

        if PipelineParameter().defectPipeline:
            defectSlices = vigra.readHDF5(inp["defect_slices"].path, 'defect_slices').astype('int64').tolist()
        else:
            defectSlices = []

        blocking = nifty.tools.blocking( roiBegin = [0L,0L,0L], roiEnd = seg.shape(), blockShape = self.blockShape )
        numberOfBlocks = blocking.numberOfBlocks
        blockOverlap = list(self.blockOverlap)

        nWorkers = min( numberOfBlocks, PipelineParameter().nThreads )
        #nWorkers = 1
        blockResult = nifty.tools.nodesToBlocksStacked(seg.get(), blocking, blockOverlap, defectSlices, nWorkers)

        blockResult = [np.array(bRes,dtype=self.dtype) for bRes in blockResult]
        self.output().writeVlen(blockResult)

    def output(self):
        save_name = "NodesToInitialBlocks.h5"
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )


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
        return { "seg" : ExternalSegmentation(self.pathToSeg), "problem" : self.problem , "nodes2blocks" : nodes2blocks }

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
        graph.deserialize( problem.read("graph") )
        numberOfEdges = graph.numberOfEdges

        global2newNodes = None if self.level == 0 else problem.read("global2new")

        workflow_logger.info("BlockwiseSubSolver: Starting extraction of subproblems.")
        t_extract = time.time()
        subproblems = self._run_subproblems_extraction(seg, graph, nodes2blocks, global2newNodes)
        workflow_logger.info( "BlockwiseSubSolver: Extraction time for subproblems %f s" % (time.time() - t_extract,) )

        workflow_logger.info("BlockwiseSubSolver: Starting solvers for subproblems.")
        t_inf_total = time.time()
        self._solve_subproblems(costs, subproblems, numberOfEdges)
        workflow_logger.info( "BlockwiseSubSolver: Inference time total for subproblems %f s" % (time.time() - t_inf_total,))

        seg.close()

    def _run_subproblems_extraction(self, seg, graph, nodes2blocks, global2newNodes):

        # get the initial blocking
        # block size in first hierarchy level
        initialBlockShape =  PipelineParameter().multicutBlockShape
        initialOverlap = list(PipelineParameter().multicutBlockOverlap)
        initialBlocking = nifty.tools.blocking( roiBegin = [0L,0L,0L], roiEnd = seg.shape(), blockShape = initialBlockShape )

        # function for subproblem extraction
        # extraction only for level 0
        def extract_subproblem(blockId, subBlocks):
            nodeList = np.unique(np.concatenate([nodes2blocks[subId] for subId in subBlocks]))
            if self.level != 0:
                nodeList = np.unique( global2newNodes[nodeList] )
            workflow_logger.debug("BlockwiseSubSolver: block id %i: Number of nodes %i" % (blockId,nodeList.shape[0]) )
            inner_edges, outer_edges, subgraph = graph.extractSubgraphFromNodes(nodeList.tolist())
            return np.array(inner_edges), np.array(outer_edges), subgraph

        blockOverlap = list(self.blockOverlap)
        blocking = nifty.tools.blocking( roiBegin = [0L,0L,0L], roiEnd = seg.shape(), blockShape = self.blockShape )
        numberOfBlocks = blocking.numberOfBlocks

        # sequential for debugging
        #subProblems = []
        #for blockId in xrange(numberOfBlocks):
        #    print "Running block:", blockId, "/", numberOfBlocks
        #    t_block = time.time()

        #    block = blocking.getBlockWithHalo(blockId, blockOverlap).outerBlock
        #    blockBegin, blockEnd = block.begin, block.end
        #    workflow_logger.debug( "Block id " + str(blockId) + " start " + str(blockBegin) + " end " + str(blockEnd) )
        #    subBlocks = initialBlocking.getBlockIdsInBoundingBox(blockBegin, blockEnd, initialOverlap)
        #    subProblems.append( extract_subproblem( blockId, subBlocks ) )

        nWorkers = min( numberOfBlocks, PipelineParameter().nThreads )
        #nWorkers = 2
        # parallel
        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for blockId in xrange(numberOfBlocks):
                block = blocking.getBlockWithHalo(blockId, blockOverlap).outerBlock
                blockBegin, blockEnd = block.begin, block.end
                subBlocks = initialBlocking.getBlockIdsInBoundingBox(blockBegin, blockEnd, initialOverlap)
                workflow_logger.debug("BlockwiseSubSolver: block id %i start %s end %s" % (blockId, str(blockBegin), str(blockEnd)) )
                tasks.append( executor.submit( extract_subproblem, blockId, subBlocks ) )
            subProblems = [task.result() for task in tasks]

        assert len(subProblems) == numberOfBlocks, str(len(subProblems)) + " , " + str(numberOfBlocks)
        return subProblems


    def _solve_subproblems(self, costs, subProblems, numberOfEdges):

        sub_solver_type = PipelineParameter().subSolverType
        if sub_solver_type in ('fm-ilp', 'fm-kl'):
            solver_params  = dict(sigma = PipelineParameter().multicutSigmaFusion,
                    number_of_iterations = PipelineParameter().multicutNumIt,
                    n_stop = PipelineParameter().multicutNumItStopGlobal,
                    n_threads = 0,
                    n_fuse    = PipelineParameter().multicutNumFuse,
                    seed_fraction = PipelineParameter().multicutSeedFraction
                    )
        else:
            solver_params = dict()

        def _solve_mc(g, costs, block_id):
            workflow_logger.debug("BlockwiseSubSolver: Solving MC Problem with %i / %i number of variables" % (g.numberOfNodes,g.numberOfEdges))
            obj = nifty.graph.optimization.multicut.multicutObjective(g, costs)
            factory = string_to_factory(obj, sub_solver_type, solver_params)
            solver = factory.create(obj)
            t_inf  = time.time()
            res    = solver.optimize()
            workflow_logger.debug("BlockwiseSubSolver: Inference for block %i with fusion moves solver in %f s" % (block_id, t_inf - time.time()) )
            return res

        # sequential for debugging
        #subResults = []
        #for blockId, subProblem in enumerate(subProblems):
        #    print "Sequential prediction for block id:", blockId
        #    subResults.append( _solve_mc( subProblem[2], costs[subProblem[0]], blockId) )

        nWorkers = min( len(subProblems), PipelineParameter().nThreads )
        #nWorkers = 1
        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = [executor.submit(
                _solve_mc,
                subProblem[2],
                costs[subProblem[0]],
                blockId) for blockId, subProblem in enumerate(subProblems)]
        subResults = [task.result() for task in tasks]

        cutEdges = np.zeros( numberOfEdges, dtype = np.uint8 )

        assert len(subResults) == len(subProblems), str(len(subResults)) + " , " + str(len(subProblems))

        for blockId in xrange(len(subProblems)):

            # get the cut edges from the subproblem
            nodeResult = subResults[blockId]
            subUvIds = subProblems[blockId][2].uvIds()

            edgeResult = nodeResult[subUvIds[:,0]] != nodeResult[subUvIds[:,1]]

            cutEdges[subProblems[blockId][0]] += edgeResult
            # add up outer edges
            cutEdges[subProblems[blockId][1]] += 1

        # all edges which are cut at least once will be cut
        cutEdges[cutEdges >= 1] = 1
        self.output().write(cutEdges)

    def output(self):
        blcksize_str = "_".join( map( str, list(self.blockShape) ) )
        save_name = "BlockwiseSubSolver_%s_%s.h5" % (blcksize_str, "modifed" if PipelineParameter().defectPipeline else "standard")
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )


# only works for level 1 for now!
class TestSubSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToClassifier = luigi.Parameter()

    blockShape   = luigi.ListParameter(default = PipelineParameter().multicutBlockShape)
    blockOverlap = luigi.ListParameter(default = PipelineParameter().multicutBlockOverlap)

    def requires(self):
        nodes2blocks = NodesToInitialBlocks(self.pathToSeg, self.blockShape, self.blockOverlap)
        return {"seg" : ExternalSegmentation(self.pathToSeg),
                "problem" : MulticutProblem(self.pathToSeg, self.pathToClassifier),
                "nodes2blocks" : nodes2blocks }

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
        graph.deserialize( problem.read("graph") )

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

        blocking = nifty.tools.blocking( roiBegin = [0L,0L,0L], roiEnd = seg.shape(), blockShape = self.blockShape )
        number_of_blocks = blocking.numberOfBlocks
        # sample block-ids corresponding to the number of threads
        n_threads = PipelineParameter().nThreads
        sampled_blocks = np.random.choice(number_of_blocks, min(n_threads, number_of_blocks), replace = False)

        # parallel
        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as executor:
            tasks = [executor.submit(extract_subproblem, block_id) for block_id in sampled_blocks]
            sub_problems = [task.result() for task in tasks]

        assert len(sub_problems) == len(sampled_blocks), "%i, %i" % (len(sub_problems), len(sampled_blocks))
        return sub_problems


    def _solve_subproblems(self, costs, sub_problems):

        def _test_mc(g, costs, sub_solver_type):

            if sub_solver_type in ('fm-ilp', 'fm-kl'):
                solver_params  = dict(sigma = PipelineParameter().multicutSigmaFusion,
                        number_of_iterations = PipelineParameter().multicutNumIt,
                        n_stop = PipelineParameter().multicutNumItStopGlobal,
                        n_threads = 0,
                        n_fuse    = PipelineParameter().multicutNumFuse,
                        seed_fraction = PipelineParameter().multicutSeedFraction
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
            workflow_logger.info( "TestSubSolver: Result of %s: mean-energy: %f, mean-inference-time: %f" % (sub_solver_type, res[0], res[1]) )
            for_serialization.append([res[0],res[1]])

        self.output().write(available, 'solvers')
        self.output().write(np.array(for_serialization), 'results')

    def output(self):
        blcksize_str = "_".join( map( str, list(self.blockShape) ) )
        save_name = "TestSubSolver_%s_%s.h5" % (blcksize_str, "modifed" if PipelineParameter().defectPipeline else "standard")
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )
