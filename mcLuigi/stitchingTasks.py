# Multicut Pipeline implemented with luigi
# Stitching Tasks

# TODO this is all WIP

import luigi

from pipelineParameter import PipelineParameter
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation
from BlockwiseMulticutSolver import BlockwiseSubSolver, NodesToInitialBlocks, fusion_moves
from multicutSolverTasks import McProblem
from customTargets import HDF5DataTarget

from tools import config_logger, config_logger

import os
import logging
import json
import time

import numpy as np
import vigra
import nifty
from itertools import combinations

from concurrent import futures

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


class BlockwiseSubSolverForStitching(luigi.Task):

    pathToSeg = luigi.Parameter()
    problem   = luigi.TaskParameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()

    def requires(self):

        nodes2blocks = NodesToInitialBlocks(self.pathToSeg, self.blockShape, self.blockOverlap)
        return { "seg" : ExternalSegmentation(self.pathToSeg), "problem" : self.problem , "nodes2blocks" : nodes2blocks }


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

        nodes2blocks = inp["nodes2blocks"].read()

        blockOverlap = list(self.blockOverlap)
        blocking = nifty.tools.blocking( roiBegin = [0L,0L,0L], roiEnd = seg.shape, blockShape = self.blockShape )
        numberOfBlocks = blocking.numberOfBlocks

        # function for subproblem extraction
        # extraction only for level 0
        def extract_subproblem(blockId, blockBegin, blockEnd):
            subBlocks = blocking.getBlockIdsInBoundingBox(blockBegin, blockEnd, blockOverlap)
            nodeList = np.unique(np.concatenate([nodes2blocks[subId] for subId in subBlocks]))
            workflow_logger.debug( "Block id %i: Number of nodes %i" % (blockId,nodeList.shape[0]) )
            inner_edges, outer_edges, subgraph = graph.extractSubgraphFromNodes(nodeList)
            return np.array(inner_edges), np.array(outer_edges), subgraph

        #nWorkers = 1
        nWorkers = min( numberOfBlocks, PipelineParameter().nThreads )

        t_extract = time.time()

        # parallel
        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for blockId in xrange(numberOfBlocks):

                block = blocking.getBlockWithHalo(blockId, blockOverlap).outerBlock
                blockBegin, blockEnd = block.begin, block.end

                workflow_logger.debug( "Block id " + str(blockId) + " start " + str(blockBegin) + " end " + str(blockEnd) )
                tasks.append( executor.submit( extract_subproblem, blockId, blockBegin, blockEnd ) )

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

        out = self.output()
        # FIXME more efficient to group these into vlen targets!
        for blockId in xrange(numberOfBlocks):
            out.write(subProblems[blockId][0],'%i_inner_edges.h5'%(blockId))
            out.write(subProblems[blockId][1],'%i_outer_edges.h5'%(blockId))
            out.write(subResults[blockId],'%i_node_results.h5'%(blockId))


    def output(self):
        blcksize_str = "_".join( map( str, list(self.blockShape) ) )
        save_name = "BlockwiseSubSolverForStitching_" + blcksize_str + ".h5"
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )


class StitchGraph(luigi.Task):

    pathToSeg = luigi.Parameter()
    mcProblem  = luigi.TaskParameter()

    def requires(self):

        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)
        blockShape = mc_config["blockShape"]
        blockOverlap = mc_config["blockOverlap"]

        nodes2blocks = NodesToInitialBlocks(self.pathToSeg, self.blockShape, self.blockOverlap)

        return {
            "mcProblem" : self.mcProblem,
            "nodes2blocks" : nodes2blocks,
            "rag" : StackedRegionAdjacencyGraph(self.pathToSeg),
            "subSolutions" : BlockwiseSubSolverForStitching(self.pathToSeg, self.mcProblem, blockShape, blockOverlap)
            }


    def run(self):

        inp = self.input()

        nodes2blocks = inp["nodes2blocks"].read()
        blockResults = inp["subSolutions"]
        rag = inp["rag"].read()

        nBlocks = len(nodes2blocks)

        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)
        blockShape = mc_config["blockShape"]
        blockOverlap = mc_config["blockOverlap"]

        blocking = nifty.blocking(roiBegin = [0L,0L,0L], roiEnd = rag.shape, blockShape = blockShape)
        assert blocking.numberOfBlocks == nBlocks

        # TODO reimplement w/o loops or parallelize subtasks

        # find all edges that are outer edges or are contained in more than one block (1)

        keepEdges = [] # these are the edges that we keep in the problem
        for blockId in xrange(nBlocks):
            keepEdges.extend(blockResult.read('%i_outer_edges'%(blockId)).tolist())

        # this looks pretty inefficient, rethink how we do this! or parallelize -> should be fine
        for blockId in xrange(nBlocks):
            thisNodes = nodes2blocks[blockId]
            block = blocking.getBlockWithHalo(blockId, blockOverlap)
            # find overlapping blocks
            overlappingBlocks = blocking.getBlockIdsInBoundingBox(block.begin, block.end, blockOverlap)
            for ovlBlockId in overlappingBlocks:
                if ovlBlockId == blockId:
                    continue
                ovlNodes = nodes2blocks[ovlBlockId]
                # find shared nodes
                sharedNodes = np.intersect1d(thisNodes,ovlNodes,assume_unique=True)
                # find shared pairs of nodes that have an edges
                for uv in combinations(sharedNodes, 2):
                    edgeId = rag.findEdge(uv[0],uv[1])
                    # TODO we might additionally check if the subsolutions agree for this edge and only
                    # add it to the keepEdges if they don't agree
                    if edgeId != -1:
                        keepEdges.append(edgeId)

        keepEdges = np.unique(keepEdges)

        # merge all other edges (only contained in a single block) according to subblock solution (2)

        fixedEdges = np.zeros(rag.numberOfEdges, dtype = bool )

        # update edges and weights (1) according to (2) (adding weights s.t. probabilities are preserved!)

        # greedy : threshold edges (1)

        # multicut : rerun multicut segmentation on (1), keeping (2) (enclosed new segments) fixed




    def output(self):



class GreedyStitching(luigi.Task):

    def requires(self):
        pass


    def run(self):
        pass


    def output(self):
        pass



class MulticutStitching(luigi.Task):

    def requires(self):
        pass


    def run(self):
        pass


    def output(self):
        pass
