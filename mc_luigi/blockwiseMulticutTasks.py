# Multicut Pipeline implemented with luigi
# Blockwise solver tasks

import luigi

from pipelineParameter import PipelineParameter
from dataTasks import ExternalSegmentation
from customTargets import HDF5DataTarget
from defectDetectionTasks import DefectSliceDetection

from tools import config_logger, run_decorator

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


def greedy(g, costs, blockId, nThreads = 0, isGlobal = False):

    assert g.numberOfEdges == costs.shape[0]
    obj = nifty.graph.multicut.multicutObjective(g, costs)

    workflow_logger.debug("Solving MC Problem greedily with %i number of variables" % (g.numberOfNodes,))

    greedy = obj.greedyAdditiveFactory().create(obj)

    solver = obj.fusionMoveBasedFactory(
        verbose=PipelineParameter().multicutVerbose,
        fusionMove=obj.fusionMoveSettings(mcFactory=ilpFac),
        proposalGen=obj.watershedProposals(sigma=PipelineParameter().multicutSigmaFusion,
            seedFraction=PipelineParameter().multicutSeedFraction),
        numberOfIterations=PipelineParameter().multicutNumIt,
        numberOfThreads=nThreads,
        numberOfParallelProposals=nParallel,
        stopIfNoImprovement=numItStop,
        fuseN=PipelineParameter().multicutNumFuse,
    ).create(obj)

    t_inf = time.time()

    # first optimize greedy
    ret = greedy.optimize()

    return ret


def fusion_moves(g, costs, blockId, nThreads = 0, isGlobal = False):

    assert g.numberOfEdges == costs.shape[0]
    obj = nifty.graph.multicut.multicutObjective(g, costs)

    workflow_logger.debug("Solving MC Problem with %i number of variables" % (g.numberOfNodes,))

    greedy = obj.greedyAdditiveFactory().create(obj)

    ilpFac = obj.multicutIlpFactory(ilpSolver=ilp_backend,verbose=0,
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True)

    if nThreads == 0:
        nParallel = 0
    else:
        nParallel = 2 * nThreads

    if isGlobal:
        numItStop = PipelineParameter().multicutNumItStopGlobal
    else:
        numItStop = PipelineParameter().multicutNumItStop

    solver = obj.fusionMoveBasedFactory(
        verbose=PipelineParameter().multicutVerbose,
        fusionMove=obj.fusionMoveSettings(mcFactory=ilpFac),
        proposalGen=obj.watershedProposals(sigma=PipelineParameter().multicutSigmaFusion,
            seedFraction=PipelineParameter().multicutSeedFraction),
        numberOfIterations=PipelineParameter().multicutNumIt,
        numberOfThreads=nThreads,
        numberOfParallelProposals=nParallel,
        stopIfNoImprovement=numItStop,
        fuseN=PipelineParameter().multicutNumFuse,
    ).create(obj)

    t_inf = time.time()

    # first optimize greedy
    ret = greedy.optimize()

    # then run the actual fusion moves solver warmstarted with greedy result

    if isGlobal: # time limit and verbose for global problem:
        visitor = obj.multicutVerboseVisitor(1,PipelineParameter().multicutGlobalTimeLimit)
        ret = solver.optimize(nodeLabels=ret,visitor=visitor)
    else:
        ret = solver.optimize(nodeLabels=ret)
    t_inf = time.time() - t_inf

    if isGlobal:
        energy = obj.evalNodeLabels(ret)
        workflow_logger.debug("Inference for global problem with fusion moves solver in " + str(t_inf) + " s")
        workflow_logger.debug("Energy of the solution: %f" % energy)
    else:
        workflow_logger.debug("Inference for block " + str(blockId) + " with fusion moves solver in " + str(t_inf) + " s")

    return ret


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

        globalProblem = problems[0]

        globalGraph = nifty.graph.UndirectedGraph()
        globalGraph.deserialize(globalProblem.read("graph"))
        globalCosts = globalProblem.read("costs")

        globalObjective = nifty.graph.multicut.multicutObjective(globalGraph, globalCosts)
        globalNumberOfNodes = globalGraph.numberOfNodes

        # we solve the problem for the costs and the edges of the last level of hierarchy
        reducedProblem = problems[-1]

        reducedGraph = nifty.graph.UndirectedGraph()
        reducedGraph.deserialize( reducedProblem.read("graph") )
        reducedCosts = reducedProblem.read("costs")

        reducedNew2Old = reducedProblem.read("new2old")
        assert len(reducedNew2Old) == reducedGraph.numberOfNodes, str(len(reducedNew2Old)) + " , " + str(reducedGraph.numberOfNodes)

        # run global inference
        t_inf = time.time()
        reducedNodeResult = fusion_moves(
                reducedGraph,
                reducedCosts,
                0,
                PipelineParameter().multicutNThreadsGlobal,
                isGlobal = True)
        workflow_logger.info("BlockwiseMulticutSolver: inference of reduced problem for the whole volume took: %f s" % (time.time() - t_inf,))

        assert reducedNodeResult.shape[0] == reducedNew2Old.shape[0]
        toGlobalNodes = reducedProblem.read("new2global")

        # TODO vectorize
        nodeResult = np.zeros(globalNumberOfNodes, dtype = 'uint32')
        for nodeId, nodeRes in enumerate(reducedNodeResult):
            nodeResult[toGlobalNodes[nodeId]] = nodeRes

        # get the global energy
        globalEnergy = globalObjective.evalNodeLabels(nodeResult)
        workflow_logger.info("BlockwiseMulticutSolver: Global problem solved with energy %f" % (globalEnergy,) )

        self.output().write(nodeResult)

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
        newCosts = np.zeros(numberOfNewEdges)
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
        self._solve_subroplems(costs, subproblems, numberOfEdges)
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


    def _solve_subroplems(self, costs, subProblems, numberOfEdges):
        # sequential for debugging
        #subResults = []
        #for blockId, subProblem in enumerate(subProblems):
        #    print "Sequential prediction for block id:", blockId
        #    subResults.append( fusion_moves( subProblem[2], costs[subProblem[0]], blockId, 1 ) )

        nWorkers = min( len(subProblems), PipelineParameter().nThreads )
        #nWorkers = 2
        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for blockId, subProblem in enumerate(subProblems):
                tasks.append( executor.submit( fusion_moves,
                    subProblem[2], costs[subProblem[0]],
                    blockId, 0 ) )
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
