# Multicut Pipeline implemented with luigi
# Blockwise solver tasks

import luigi

from pipelineParameter import PipelineParameter
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation
from multicutSolverTasks import McProblem#, McSolverFusionMoves, MCSSolverOpengmExact
from customTargets import HDF5DataTarget

from tools import config_logger

import os
import logging
import json
import time

import numpy as np
import vigra
import nifty

from concurrent import futures

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


def fusion_moves(g, costs, blockId, nThreads = 1):

    # read the mc parameter
    with open(PipelineParameter().MCConfigFile, 'r') as f:
        mc_config = json.load(f)

    assert g.numberOfEdges == costs.shape[0]

    obj = nifty.graph.multicut.multicutObjective(g, costs)

    workflow_logger.debug("Solving MC Problem with %i number of variables" % (g.numberOfNodes,))

    greedy = obj.greedyAdditiveFactory().create(obj)

    ilpFac = obj.multicutIlpFactory(ilpSolver='cplex',verbose=0,
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True)

    if nThreads == 1:
        nParallel = 1
    else:
        nParallel = 2 * nThreads

    solver = obj.fusionMoveBasedFactory(
        verbose=mc_config["verbose"],
        fusionMove=obj.fusionMoveSettings(mcFactory=ilpFac),
        proposalGen=obj.watershedProposals(sigma=mc_config["sigmaFusion"],seedFraction=mc_config["seedFraction"]),
        numberOfIterations=mc_config["numIt"],
        numberOfThreads=nThreads,
        numberOfParallelProposals=nParallel,
        stopIfNoImprovement=mc_config["numItStop"],
        fuseN=mc_config["numFuse"],
    ).create(obj)

    t_inf = time.time()

    # first optimize greedy
    ret = greedy.optimize()

    # then run the actual fusion moves solver warmstarted with greedy result
    # TODO time limit
    ret = solver.optimize(nodeLabels=ret)
    t_inf = time.time() - t_inf

    workflow_logger.debug("Inference for block " + str(blockId) + " with fusion moves solver in " + str(t_inf) + " s")

    return ret


# produce reduced global graph from subproblems
# solve global multicut problem on the reduced graph
class BlockwiseMulticutSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    globalProblem  = luigi.TaskParameter()

    numberOflevels = luigi.Parameter()

    def requires(self):
        # read the mc parameter
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        # block size in first hierarchy level
        initialBlockShape = mc_config["blockShape"]
        # block overlap, for now same for each hierarchy lvl
        blockOverlap = mc_config["blockOverlap"]

        problemHierarchy = [self.globalProblem,]
        blockFactor = 1

        for l in xrange(self.numberOflevels):
            levelBlockShape = map( lambda x: x*blockFactor, initialBlockShape)

            workflow_logger.info("Scheduling reduced problem for level " + str(l) + " with block shape: " + str(levelBlockShape))

            # TODO check that we don't get larger than the actual shape here
            problemHierarchy.append( ReducedProblem(self.pathToSeg, problemHierarchy[-1],
                levelBlockShape , blockOverlap, l))
            blockFactor *= 2

        return problemHierarchy


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

        # FIXME parallelism makes it slower here -> investigate this further and discuss with thorsten!
        t_inf = time.time()
        reducedNodeResult = fusion_moves( reducedGraph, reducedCosts, "reduced global", 20 )
        t_inf = time.time() - t_inf
        workflow_logger.info("Inference of reduced problem for the whole volume took: %f s" % (t_inf,))

        assert reducedNodeResult.shape[0] == reducedNew2Old.shape[0]

        # TODO clean up and vectorize some
        # project back to global problem through all hierarchy levels
        nodeResult = reducedNodeResult
        new2old    = reducedNew2Old
        for l in reversed(xrange(self.numberOflevels)):

            nextProblem = problems[l]
            if l != 0:
                nextNew2old = nextProblem.read("new2old")
                nextNumberOfNodes = len( nextNew2old )
            else:
                nextNumberOfNodes = globalNumberOfNodes

            nextNodeResult = np.zeros(nextNumberOfNodes, dtype = 'uint32')
            for nodeId in xrange(nodeResult.shape[0]):
                for oldNodeId in new2old[nodeId]:
                    nextNodeResult[oldNodeId] = nodeResult[nodeId]

            nodeResult = nextNodeResult
            if l != 0:
                new2old = nextNew2old

        # get the global energy
        globalEnergy = globalObjective.evalNodeLabels(nodeResult)
        workflow_logger.info("Blockwise Multicut problem solved with energy %s" % (globalEnergy,) )

        self.output().write(nodeResult)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "BlockwiseMulticutSolver.h5")
        return HDF5DataTarget( save_path )


class ReducedProblem(luigi.Task):

    pathToSeg = luigi.Parameter()
    problem   = luigi.TaskParameter()

    blockShape     = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()

    level = luigi.Parameter()

    def requires(self):
        return {"subSolution" : BlockwiseSubSolver( self.pathToSeg, self.problem, self.blockShape, self.blockOverlap, self.level),
                "problem" : self.problem }


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
        ufd = nifty.ufd.Ufd( numberOfNodes )

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

        else:
            global2newLast = problem.read("global2new").astype(np.uint32)
            global2new = np.zeros_like( global2newLast, dtype = np.uint32)

            #TODO vectorize  completely
            for newNode in xrange(numberOfNewNodes):
                global2new[ global2newLast[new2oldNodes[newNode]] ] = newNode


        t_merge = time.time() - t_merge
        workflow_logger.info("Time for merging: %f s" % (t_merge))

        workflow_logger.info("Merging of blockwise results reduced problemsize (level %i):" % (self.level,) )
        workflow_logger.info("Nodes: From %i to %i" % (numberOfNodes, numberOfNewNodes) )
        workflow_logger.info("Edges: From %i to %i" % (numberOfEdges, numberOfNewEdges) )

        out = self.output()
        out.write(reducedGraph.serialize(), "graph")
        out.write(newCosts, "costs")
        out.write(global2new, "global2new")
        # need to serialize this differently, due to list of list
        new2oldNodes = np.array([np.array(n2o) for n2o in new2oldNodes])
        out.writeVlen(new2oldNodes, "new2old")


    def output(self):
        blcksize_str = "_".join( map( str, list(self.blockShape) ) )
        save_name = "ReducedProblem_" + blcksize_str + ".h5"
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )


class NodesToInitialBlocks(luigi.Task):

    pathToSeg = luigi.Parameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()

    def requires(self):
        return ExternalSegmentation(self.pathToSeg)

    def run(self):

        seg = self.input()
        seg.open()


        blocking = nifty.tools.blocking( roiBegin = [0L,0L,0L], roiEnd = seg.shape, blockShape = self.blockShape )
        numberOfBlocks = blocking.numberOfBlocks
        blockOverlap = list(self.blockOverlap)

        def nodes_to_block(blockId):
            block = blocking.getBlockWithHalo(blockId, blockOverlap).outerBlock
            blockBegin, blockEnd = block.begin, block.end
            return np.unique( seg.read(blockBegin, blockEnd) )

        #nWorkers = 1
        nWorkers = min( numberOfBlocks, PipelineParameter().nThreads )

        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for blockId in xrange(numberOfBlocks):
                tasks.append( executor.submit(nodes_to_block, blockId) )
            blockResult = np.array([fut.result() for fut in tasks])

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

        #with open(PipelineParameter().MCConfigFile, 'r') as f:
        #    mc_config = json.load(f)
        #initialShape = mc_config["blockShape"]
        #overlap      = mc_config["blockOverlap"]

        #nodes2blocks = NodesToInitialBlocks(self.pathToSeg, initialShape, overlap)
        return { "seg" : ExternalSegmentation(self.pathToSeg), "problem" : self.problem }#, "nodes2blocks" : nodes2blocks }


    def run(self):

        # Input
        inp = self.input()
        seg = inp["seg"]
        seg.open()

        problem = inp["problem"]
        costs = problem.read("costs")

        graph = nifty.graph.UndirectedGraph()
        graph.deserialize( problem.read("graph") )
        numberOfEdges = graph.numberOfEdges

        if self.level == 0:
            global2newNodes = None
        else:
            global2newNodes = problem.read("global2new")

        # function for subproblem extraction
        # extraction for every level
        def extract_subproblem(blockBegin, blockEnd):
            nodeList = np.unique( seg.read(blockBegin, blockEnd) )
            if self.level != 0:
                nodeList = np.unique( global2newNodes[nodeList] )

            inner_edges, outer_edges, subgraph = graph.extractSubgraphFromNodes(nodeList)
            return np.array(inner_edges), np.array(outer_edges), subgraph


        # TODO test this
        #nodes2blocks = inp["nodes2blocks"].read()

        ## get the initial blocking
        #with open(PipelineParameter().MCConfigFile, 'r') as f:
        #    mc_config = json.load(f)
        ## block size in first hierarchy level
        #initialBlockShape = mc_config["blockShape"]
        #initialOverlap = list(mc_config["blockOverlap"])
        #initialBlocking = nifty.tools.blocking( roiBegin = [0L,0L,0L], roiEnd = seg.shape, blockShape = initialBlockShape )

        ## function for subproblem extraction
        ## extraction only for level 0
        #def extract_subproblem(blockBegin, blockEnd):
        #    subBlocks = initialBlocking.getBlockIdsInBoundingBox(blockBegin, blockEnd, initialOverlap)
        #    nodeList = np.unique(np.concatenate([nodes2blocks[subId] for subId in subBlocks]))
        #    inner_edges, outer_edges, subgraph = graph.extractSubgraphFromNodes(nodeList)
        #    return np.array(inner_edges), np.array(outer_edges), subgraph


        blockOverlap = list(self.blockOverlap)
        blocking = nifty.tools.blocking( roiBegin = [0L,0L,0L], roiEnd = seg.shape, blockShape = self.blockShape )
        numberOfBlocks = blocking.numberOfBlocks

        #nWorkers = 1
        nWorkers = min( numberOfBlocks, PipelineParameter().nThreads )

        t_extract = time.time()

        # sequential for debugging
        #subProblems = []
        #for blockId in xrange(numberOfBlocks):

        #    # nifty blocking
        #    block = blocking.getBlockWithHalo(blockId, self.blockOverlap).outerBlock
        #    blockBegin, blockEnd = block.begin, block.end

        #    subProblems.append( extract_subproblem( blockBegin, blockEnd ) )

        # parallel
        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for blockId in xrange(numberOfBlocks):

                block = blocking.getBlockWithHalo(blockId, blockOverlap).outerBlock
                blockBegin, blockEnd = block.begin, block.end

                workflow_logger.debug( "Block id " + str(blockId) + " start " + str(blockBegin) + " end " + str(blockEnd) )
                tasks.append( executor.submit( extract_subproblem, blockBegin, blockEnd ) )

            subProblems = [task.result() for task in tasks]

        assert len(subProblems) == numberOfBlocks, str(len(subProblems)) + " , " + str(numberOfBlocks)

        t_extract = time.time() - t_extract
        workflow_logger.info( "Extraction time for subproblems %f s" % (t_extract,) )

        t_inf_total = time.time()

        # sequential for debugging
        #subResults = []
        #for blockId, subProblem in enumerate(subProblems):
        #    subResults.append( fusion_moves( subProblem[2], costs[subProblem[0]], blockId, 1 ) )

        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for blockId, subProblem in enumerate(subProblems):
                tasks.append( executor.submit( fusion_moves, subProblem[2],
                    costs[subProblem[0]], blockId, 1 ) )
        subResults = [task.result() for task in tasks]

        t_inf_total = time.time() - t_inf_total
        workflow_logger.info( "Inference time total for subproblems %f s" % (t_inf_total,))

        cutEdges = np.zeros( numberOfEdges, dtype = np.uint8 )

        assert len(subResults) == len(subProblems), str(len(subResults)) + " , " + str(len(subProblems))

        for blockId in xrange(numberOfBlocks):

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
        save_name = "BlockwiseSubSolver_" + blcksize_str + ".h5"
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )
