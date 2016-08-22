# Multicut Pipeline implemented with luigi
# Blockwise solver tasks

import luigi

from pipelineParameter import PipelineParameter
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation
from multicutSolverTasks import McProblem#, McSolverFusionMoves, MCSSolverOpengmExact
from customTargets import HDF5DataTarget

from toolsLuigi import UnionFind, config_logger, get_blocks

import os
import logging
import json
import time

import numpy as np
import vigra
import nifty

from concurrent import futures

# TODO would be better to let the scheduler handle the parallelisation
# -> start corresponding MCProblem / MCSolver subtasks in parallel
# once we know how this works...

# so this is only preliminary
# -> need more tasks and more atomicity!

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


# serialization and deserialization for list of lists with hdf5

def serializeNew2Old(out, new2old):
    # we can't serialize the whole list of list, so we do it in this ugly manner...
    for i, nList in enumerate(new2old):
        out.write(nList, "new2old/" + str(i))


def deserializeNew2Old(inp):
    new2old = []
    nodeIndex = 0
    while True:
        try:
            new2old.append( inp.read("new2old/" + str(nodeIndex) ) )
            nodeIndex += 1
        except KeyError:
            break
    return new2old



# TODO use Fusionmoves task instead
def fusion_moves(g, costs, blockId, parallelProposals):

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

    solver = obj.fusionMoveBasedFactory(
        verbose=mc_config["verbose"],
        fusionMove=obj.fusionMoveSettings(mcFactory=ilpFac),
        proposalGen=obj.watershedProposals(sigma=mc_config["sigmaFusion"],seedFraction=mc_config["seedFraction"]),
        numberOfIterations=mc_config["numIt"],
        numberOfParallelProposals=parallelProposals*2, # we use the parameter n parallel instead of the config here
        numberOfThreads=parallelProposals,
        stopIfNoImprovement=mc_config["numItStop"],
        fuseN=mc_config["numFuse"],
    ).create(obj)

    t_inf = time.time()
    ret = greedy.optimize()

    # test time limit
    #visitor = obj.multicutVerboseVisitor(100, 10) # print, timeLimit
    #ret = solver.optimize(nodeLabels=ret, visitor = visitor)

    #ret = solver.optimize(nodeLabels=ret)
    t_inf = time.time() - t_inf

    workflow_logger.debug("Inference for block " + str(blockId) + " with fusion moves solver in " + str(t_inf) + " s")

    return ret


# produce reduced global graph from subproblems
# solve global multicut problem on the reduced graph
class BlockwiseMulticutSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToRF  = luigi.Parameter()

    numberOflevels = luigi.Parameter()
    #initialBlocksize = luigi.Parameter()

    def requires(self):
        globalProblem = McProblem( self.pathToSeg, self.pathToRF)

        # for now, we hardcode the block overlaps here and don't change it throughout the different hierarchy levels.
        # in the end, this sould also be a parameter (and maybe even change in the different hierarchy levels)
        blockOverlap = [4,20,20]
        # nested block hierarchy TODO make it possible to set this more explicit.

        # 1st hierarchy level: 50 x 512 x 512 blocks (hardcoded for now)
        initialBlockSize = [50,512,512]

        problemHierarchy = [globalProblem,]
        blockFactor = 1
        for l in xrange(self.numberOflevels):
            levelBlocksize = map( lambda x: x*blockFactor, initialBlockSize)
            workflow_logger.info("Scheduling reduced problem for level " + str(l) + " with block size: " + str(levelBlocksize))
            # TODO check that we don't get larger than the actual shape here
            problemHierarchy.append( ReducedProblem(self.pathToSeg, problemHierarchy[-1],
                levelBlocksize , blockOverlap, l) )
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

        reducedNew2Old = deserializeNew2Old(reducedProblem)
        assert len(reducedNew2Old) == reducedGraph.numberOfNodes, str(len(reducedNew2Old)) + " , " + str(reducedGraph.numberOfNodes)

        # FIXME parallelism makes it slower here -> investigate this further and discuss with thorsten!
        t_inf = time.time()
        #reducedNodeResult = fusion_moves( reducedGraph, reducedCosts, "reduced global", 20 )
        reducedNodeResult = fusion_moves( reducedGraph, reducedCosts, "reduced global", 1 )
        t_inf = time.time() - t_inf
        workflow_logger.info("Inference of reduced problem for the whole volume took: %f s" % (t_inf,))

        assert reducedNodeResult.shape[0] == len(reducedNew2Old)

        # project back to global problem through all hierarchy levels
        nodeResult = reducedNodeResult
        problem    = reducedProblem
        new2old    = reducedNew2Old
        for l in reversed(xrange(self.numberOflevels)):

            nextProblem = problems[l]
            if l != 0:
                nextNew2old = deserializeNew2Old( nextProblem )
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

    blockSize     = luigi.ListParameter()
    blockOverlaps = luigi.ListParameter()

    level = luigi.Parameter()

    def requires(self):
        return {"subSolution" : BlockwiseSubSolver( self.pathToSeg, self.problem, self.blockSize, self.blockOverlaps, self.level ),
                "problem" : self.problem }


    def run(self):

        inp = self.input()
        problem   = inp["problem"]
        cutEdges = inp["subSolution"].read()

        g = nifty.graph.UndirectedGraph()
        g.deserialize(problem.read("graph"))

        numberOfNodes = g.numberOfNodes
        numberOfEdges = g.numberOfEdges

        uvIds = g.uvIds()#.astype('uint32')
        costs  = problem.read("costs")

        t_merge = time.time()
        ufd = nifty.ufd.Ufd( numberOfNodes )

        mergeNodes = uvIds[cutEdges == 0]

        for uv in mergeNodes:
            ufd.merge(int(uv[0]), int(uv[1]))

        old2newNodes = ufd.elementLabeling()
        new2oldNodes = ufd.representativesToSets()
        # number of nodes for the new problem
        numberOfNewNodes = len(new2oldNodes)

        # find new edges and new edge weights
        activeEdges = np.where( cutEdges == 1 )[0]
        newEdges = {}
        for edgeId in activeEdges:
            node0 = old2newNodes[uvIds[edgeId,0]]
            node1 = old2newNodes[uvIds[edgeId,1]]
            # we have to be in different new nodes!
            assert node0 != node1, str(node0) + " , " + str(node1) + " @ edge: " + str(edgeId)
            uNew = min(node0, node1)
            vNew = max(node0, node1)
            if (uNew, vNew) in newEdges:
                newEdges[(uNew,vNew)] += costs[edgeId]
            else:
                newEdges[(uNew,vNew)]  = costs[edgeId]

        numberOfNewEdges = len( newEdges )
        uvIdsNew = np.array( newEdges.keys() )

        reducedGraph = nifty.graph.UndirectedGraph(numberOfNewNodes)
        reducedGraph.insertEdges(uvIdsNew)

        assert uvIdsNew.shape[0] == numberOfNewEdges, str(uvIdsNew.shape[0]) + " , " + str(numberOfNewEdges)
        # this should have the correct order
        newCosts = np.array( newEdges.values() )

        if self.level == 0:
            global2new = old2newNodes

        else:
            global2newLast = problem.read("global2new")
            global2new = np.zeros_like( global2newLast, dtype = np.uint32)
            for newNode in xrange(numberOfNewNodes):
                for oldNode in new2oldNodes[newNode]:
                    global2new[global2newLast[oldNode]] = newNode

        t_merge = time.time() - t_merge
        workflow_logger.info("Time for merging: %f s" % (t_merge))

        workflow_logger.info("Merging of blockwise results reduced problemsize:" )
        workflow_logger.info("Nodes: From %i to %i" % (numberOfNodes, numberOfNewNodes) )
        workflow_logger.info("Edges: From %i to %i" % (numberOfEdges, numberOfNewEdges) )

        out = self.output()
        out.write(reducedGraph.serialize(), "graph")
        out.write(newCosts, "costs")
        out.write(global2new, "global2new")
        # need to serialize this differently, because hdf5 can't natively save lists of lists
        serializeNew2Old(out, new2oldNodes)


    def output(self):
        blcksize_str = "_".join( map( str, list(self.blockSize) ) )
        save_name = "ReducedProblem_" + blcksize_str + ".h5"
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )



class BlockwiseSubSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    problem   = luigi.TaskParameter()

    blockSize     = luigi.ListParameter()
    blockOverlaps = luigi.ListParameter()

    level = luigi.Parameter()

    def requires(self):
        return { "seg" : ExternalSegmentation(self.pathToSeg), "problem" : self.problem }

    def run(self):

        def extract_subproblems(globalSegmentation, globalGraph, blockBegin, blockEnd,  global2new):
            # TODO better to implement this in cpp ?! -> nifty::hdf5
            nodeList = np.unique( globalSegmentation.read(blockBegin, blockEnd) )
            if self.level != 0:
                # TODO no for loop !
                for i in xrange(nodeList.shape[0]):
                    nodeList[i] = global2new[nodeList[i]]
                nodeList = np.unique(nodeList)
            return nifty.graph.extractSubgraphFromNodes(globalGraph, nodeList)

        #def extract_subproblems(globalGraph, blockBegin, blockEnd,  global2new):
        #    return nifty.graph.extractSubgraphFromNodes(globalGraph, nodeList, global2new)

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
            # TODO this needs to be the identity for level 0
            global2newNodes = None
        else:
            global2newNodes = problem.read("global2new")

        # TODO this function is implemented VERY ugly, best to replace this with vigra or nifty functionality
        numberOfBlocks, blockBegins, blockEnds = get_blocks(seg.shape, self.blockSize, self.blockOverlaps)

        #nWorkers = 1
        nWorkers = min( numberOfBlocks, PipelineParameter().nThreads )

        t_extract = time.time()

        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for blockId in xrange(numberOfBlocks):
                workflow_logger.debug( "Block id " + str(blockId) + " start " + str(blockBegins[blockId]) + " end " + str(blockEnds[blockId]) )
                tasks.append( executor.submit( extract_subproblems, seg, graph, blockBegins[blockId], blockEnds[blockId], global2newNodes ) )

        subProblems = [task.result() for task in tasks]

        assert len(subProblems) == numberOfBlocks, str(len(subProblems)) + " , " + str(numberOfBlocks)

        t_extract = time.time() - t_extract
        workflow_logger.info( "Extraction time for subproblems %f s" % (t_extract,) )

        t_inf_total = time.time()

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

            ru = nodeResult[subUvIds[:,0]]
            rv = nodeResult[subUvIds[:,1]]
            edgeResult = ru!=rv

            # add up cut inner edges
            cutEdges[subProblems[blockId][0]] += edgeResult

            # add up outer edges
            cutEdges[subProblems[blockId][1]] += 1

        # all edges which are cut at least once will be cut
        cutEdges[cutEdges >= 1] = 1

        self.output().write(cutEdges)


    def output(self):
        blcksize_str = "_".join( map( str, list(self.blockSize) ) )
        save_name = "BlockwiseSubSolver_" + blcksize_str + ".h5"
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )
